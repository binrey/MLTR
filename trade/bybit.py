from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import sleep

import pandas as pd
from loguru import logger

from common.type import Side
from common.utils import Telebot, date2name, plot_fig
from trade.utils import Position

pd.options.mode.chained_assignment = None
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import stackprinter
import yaml
from easydict import EasyDict
from pybit.unified_trading import HTTP, WebSocket

from common.utils import PyConfig, date2str
from experts import ByBitExpert

stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen
 
def get_bybit_hist(mresult, size):
    data = EasyDict(Date=np.empty(size, dtype=np.datetime64),
        Id=np.zeros(size, dtype=np.int64),
        Open=np.zeros(size, dtype=np.float32),
        Close=np.zeros(size, dtype=np.float32),
        High=np.zeros(size, dtype=np.float32),
        Low=np.zeros(size, dtype=np.float32),
        Volume=np.zeros(size, dtype=np.int64)
        )    

    input = np.array(mresult["list"], dtype=np.float64)[::-1]
    data.Id = input[:, 0].astype(np.int64)
    data.Date = data.Id.astype("datetime64[ms]")
    data.Open = input[:, 1]
    data.High = input[:, 2]
    data.Low  = input[:, 3]
    data.Close= input[:, 4]
    data.Volume = input[:, 5]
    return data


@dataclass
class StepData:
    curr: Any = None
    prev: Any = None
    
    def update(self, curr_value: Any):
        self.prev = deepcopy(self.curr)
        self.curr = curr_value
        
    def change(self):
        return str(self.curr) != str(self.prev)

class BybitTrading:
    def __init__(self, cfg, credentials) -> None:
        self.cfg = cfg
        self.time_rounded = 0
        self.my_telebot = Telebot(credentials["bot_token"])
        self.h, self.time = None, StepData()
        self.pos: StepData[Position] = StepData()
        self.session = HTTP(
            testnet=False,
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
        )
        self.exp = ByBitExpert(self.cfg, self.session)
            
        self.hist2plot, self.lines2plot, self.sl = None, None, None
        
        self.save_path = Path("real_trading") / f"{self.cfg.ticker}-{self.cfg.period}"
        self.backup_path = self.save_path / "backup.pkl"

    def clear_log_dir(self):
        if self.cfg.save_plots:
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

    def save_backup(self):
        backup_data = {
            "hist2plot": self.hist2plot,
            "lines2plot": self.lines2plot,
            "sl": self.sl,
            "open_position": self.pos
        }
        backup_path = self.save_path / "backup.pkl"
        with open(backup_path, "wb") as f:
            pickle.dump(backup_data, f)
        logger.debug(f"backup saved to {backup_path}")

    def load_backup(self):
        if self.backup_path.exists():
            with open(self.backup_path, "rb") as f:
                backup_data = pickle.load(f)
            self.hist2plot = backup_data["hist2plot"]
            self.lines2plot = backup_data["lines2plot"]
            self.sl = backup_data["sl"]
            self.pos = backup_data["open_position"]
            logger.info(f"backup loaded from {self.backup_path}")
        else:
            logger.info("no backup found")

    def test_connection(self):
        self.update_market_state()
        if self.pos.curr is not None:
            self.exp.active_position = self.pos.curr
            self.load_backup()
        else:
            self.clear_log_dir()
            
    def handle_trade_message(self, message):
        # try:
        data = message.get('data')
        self.time.update(self.to_datetime(data[0].get("T")))
        time_rounded = int(int(data[0].get("T"))/1000/60/int(self.cfg.period[1:]))
        print ("\033[A\033[A")
        logger.info(f"server time: {date2str(self.time.curr)}")
        # except (ValueError, AttributeError):
            # pass            
        if time_rounded > self.time_rounded:
            if self.time_rounded:
                self.update()
                actpos = f"{self.pos.curr.ticker} {self.pos.curr.side} {self.pos.curr.volume}" if self.pos.curr is not None else "пусто"
                msg = f"{date2str(self.time.curr)}: cur. pos: {actpos}"
                logger.info(msg)
                print()
                self.my_telebot.send_text(msg)
            self.time_rounded = time_rounded

    def trailing_sl(self):
        if self.h is None or self.pos.curr is None:
            return
        sl = float(self.pos.curr.sl)
        sl_new = float(sl + self.cfg.trailing_stop_rate*(self.h.Open[-1] - sl))
        sl_new = sl_new if abs(sl_new - self.h.Open[-1]) - self.cfg.ticksize > 0 else sl
        self.update_trailing_stop(sl_new)
        return float(sl)       

    def update_trailing_stop(self, sl_new):
        try:
            resp = self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg.ticker,
                stopLoss=sl_new,
                slTriggerB="IndexPrice",
                positionIdx=0,
            )
            logger.debug(f"trailing sl: {self.pos.curr.sl:.2f} -> {sl_new:.2f}")
        except Exception as ex:
            logger.error(ex)


    def to_datetime(self, timestamp: Union[int, float]) -> np.datetime64:
        return np.datetime64(int(timestamp), "ms")
    
    def get_pos_hist(self, limit=1):
        positions = self.session.get_closed_pnl(category="linear", limit=limit)["result"]["list"]
        positions = [self.build_position(pos) for pos in positions if pos["symbol"] == self.cfg.ticker]
        return positions

    def update_market_state(self):
            self.open_orders = self.session.get_open_orders(category="linear", symbol=self.cfg.ticker)["result"]["list"]
            positions = self.session.get_positions(category="linear", symbol=self.cfg.ticker)["result"]["list"]
            positions = [pos for pos in positions if pos["size"] != "0"]
            if len(positions):
                self.pos.update(self.build_position(positions[0]))
            else:
                self.pos.update(None)
                
    def build_position(self, pos: Dict[str, Any]):
        return Position(
            price=float(pos["avgPrice"])*Side.from_str(pos["side"]).value,
            date=self.to_datetime(pos["updatedTime"]),
            indx=0,
            ticker=pos["symbol"],
            volume=float(pos["size"]),
            period=self.cfg.period,
            sl=float(pos["stopLoss"])
            )            

    def update_hist(self):
        message = self.session.get_kline(
            category="linear",
            symbol=self.cfg.ticker,
            interval=self.cfg.period[1:],
            start=0,
            end=self.time.curr,
            limit=self.cfg.hist_buffer_size
        )
        return get_bybit_hist(message["result"], self.cfg.hist_buffer_size)

    def update(self):
        self.update_market_state()    
                
        self.sl = self.trailing_sl()

        self.h =self.update_hist()

        if self.cfg.save_plots:
            if self.pos.change(): 
                if self.pos.prev is None:
                    self.hist2plot = pd.DataFrame(self.h)
                    self.hist2plot.set_index(pd.to_datetime(self.hist2plot.Date), drop=True, inplace=True)
                    self.lines2plot = deepcopy(self.exp.lines)
                    for line in self.lines2plot:
                        for i, point in enumerate(line):
                            y = point[1]
                            try:
                                y = y.item() #  If y is 1D numpy array
                            except:
                                pass
                            x = point[0]
                            x = max(self.hist2plot.Id.iloc[0], x)
                            x = min(self.hist2plot.Id.iloc[-1], x)
                            line[i] = (self.hist2plot.index[self.hist2plot.Id==x][0], y)    
                    open_time = pd.to_datetime(self.pos.curr.open_date.astype("datetime64[m]"))
                    side = str(self.pos.curr.side)
                    self.lines2plot.append([(open_time, self.pos.curr.open_price), (open_time, self.pos.curr.open_price)])
                    self.lines2plot.append([(open_time, self.pos.curr.sl), (open_time, self.pos.curr.sl)])
                    # plot_fig(self.hist2plot, self.lines2plot, self.save_path, None, open_time, side, self.cfg.ticker)
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, open_time, side, self.cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2name(open_time))
                    self.hist2plot = self.hist2plot.iloc[:-1]                    
                else:
                    last_row = pd.DataFrame(self.h).iloc[-2:]
                    last_row.index = pd.to_datetime(last_row.Date)
                    self.hist2plot = pd.concat([self.hist2plot, last_row])                    
                    self.lines2plot[-2][-1] = (pd.to_datetime(self.hist2plot.iloc[-1].Date), self.lines2plot[-2][-1][-1])
                    self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
                    open_time = pd.to_datetime(self.time.prev)
                    side = str(self.pos.prev.side)
                    # plot_fig(self.hist2plot, self.lines2plot, self.save_path, None, open_time, side, self.cfg.ticker)
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, open_time, side, self.cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2name(open_time))
                    self.hist2plot = None    
            elif self.pos.curr is not None:
                h = pd.DataFrame(self.h).iloc[-2:-1]
                h.set_index(pd.to_datetime(h.Date), drop=True, inplace=True)
                self.hist2plot = pd.concat([self.hist2plot, h])
                self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
        
        texp = self.exp.update(self.h, self.pos.curr)
        self.save_backup()
        

def launch(cfg, demo=False):
    with open("./api.yaml", "r") as f:
        creds = yaml.safe_load(f)
    if demo:
        creds["api_secret"] = creds["api_secret_demo"]
        creds["api_key"] = creds["api_key_demo"]
    
    public = WebSocket(channel_type='linear', testnet=False)
    bybit_trading = BybitTrading(cfg, creds)
    bybit_trading.test_connection()
    public.trade_stream(symbol=cfg.ticker, callback=bybit_trading.handle_trade_message)
    
    print()
    while True:
        sleep(60)
        if not public.is_connected():
            logger.warning("Connection is lost! Reconnect...")
            public.exit()
            public = WebSocket(channel_type='linear', testnet=False)
            public.trade_stream(symbol=cfg.ticker, callback=bybit_trading.handle_trade_message)
            sleep(1)
            msg = f"Request connection status: is_connected={public.is_connected()}\n"
            logger.warning(msg)
            bybit_trading.my_telebot.send_text(msg)
    

    
    