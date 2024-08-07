from pathlib import Path
from shutil import rmtree
from time import sleep
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
pd.options.mode.chained_assignment = None
from experts import ByBitExpert
from utils import PyConfig
from datetime import datetime
import numpy as np
import pandas as pd
from easydict import EasyDict
from copy import deepcopy
import telebot
import yaml
from PIL import Image
import pickle
from multiprocessing import Process
import stackprinter
from pybit.unified_trading import HTTP, WebSocket
from backtest_broker import Position
from typing import Optional
from collections import defaultdict


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
    data.Date = data.Id*1000000
    data.Open = input[:, 1]
    data.High = input[:, 2]
    data.Low  = input[:, 3]
    data.Close= input[:, 4]
    data.Volume = input[:, 5]
    return data
                
             
class Telebot:
    def __init__(self, token) -> None:
        self.bot = telebot.TeleBot(token)
        self.chat_id = 480902846
    def send_image(self, img_path, caption=None):
        try:
            img = Image.open(img_path)
            if caption is not None:
                self.bot.send_message(self.chat_id, caption)
            self.bot.send_photo(self.chat_id, img)
        except Exception as ex:
            self.bot.send_message(self.chat_id, ex)

    def send_text(self, text):
            self.bot.send_message(self.chat_id, text)        
                
                
def date2save_format(date, prefix=None):
    s = str(np.array(date).astype('datetime64[s]')).split('.')[0].replace(":", "-") 
    if prefix is not None and len(prefix) > 0:
        s += f"-{prefix}"
    return s + ".png"

     
def plot_fig(hist2plot, lines2plot, save_path=None, prefix=None, t=None, side=None, ticker="X"):
    lines_in_range = [[] for _ in range(len(lines2plot))]
    for i, line in enumerate(lines2plot):
        assert len(line) >= 2, "line must have more than 1 point"
        for point_id, point in enumerate(line):
            assert len(point) == 2
            point = (point[0], float(point[1]))
            assert type(point[0]) is pd.Timestamp, f"point[0]={point[0]} in line {i}, must be pd.Timestamp, but has type {type(point[0])}"
            assert type(point[1]) is float, f"point[1]={point[1]} in line {i}, must be float, but has type {type(point[1])}"
            if point[0] >= hist2plot.index[0]:
                lines_in_range[i].append(point)
            elif len(line) == 2:
                lines_in_range[i].append((hist2plot.index[-1], point[1]))
                
            assert point[0] <= hist2plot.index[-1]
    mystyle=mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
    kwargs = dict(
        type='candle', 
        block=False,
        alines=dict(alines=lines_in_range, linewidths=[1]*len(lines2plot)),
        volume=True,
        figscale=1.5,
        style=mystyle,
        datetime_format='%m-%d %H:%M:%Y',
        title=f"{np.array(t).astype('datetime64[s]')}-{ticker}-{side}",
        returnfig=True
    )

    fig, axlist = mpf.plot(data=hist2plot, **kwargs)
    
    if side.lower() in ["buy", "sell"]:
        side_int = 1 if side.lower() == "buy" else -1
        x = hist2plot.index.get_loc(t)
        if type(x) is slice:
            x = x.start
        y = hist2plot.loc[t].Open
        if y.ndim > 0:
            y = y.iloc[0]
        arrow_size = (hist2plot.iloc[-10:].High - hist2plot.iloc[-10:].Low).mean()
        axlist[0].annotate("", (x, y + arrow_size*side_int), fontsize=20, xytext=(x, y),
                    color="black", 
                    arrowprops=dict(
                        arrowstyle='->',
                        facecolor='b',
                        edgecolor='b'))
        
    if save_path is not None:
        save_path = save_path / date2save_format(t, prefix)  
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.2)  
    plt.close('all')
    return save_path
      
      
def log_position(t, hist2plot, lines2plot, save_path):
    d = {"time": t,
         "hist": hist2plot,
         "lines": lines2plot
         }
    save_path = save_path / f"{str(t).split('.')[0]}.pkl".replace(":", "-")
    pickle.dump(d, open(save_path, "wb"))
    logger.info(f"save logs to {save_path}")
    

class BybitTrading:
    def __init__(self, cfg, credentials) -> None:
        self.cfg = cfg
        self.t0 = 0
        self.my_telebot = Telebot(credentials["bot_token"])
        self.h = None
        self.open_position: Optional[Position] = None
        self.session = HTTP(
            testnet=False,
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
        )
        self.exp = ByBitExpert(cfg, self.session)
        if cfg.save_plots:
            self.save_path = Path("real_trading") / f"{cfg.ticker}-{cfg.period}"
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir()
            
        self.hist2plot, self.lines2plot, self.open_time, self.side, self.sl = None, None, None, None, None
        
    def test_connection(self):
        self.get_open_orders_positions()        
        if self.open_position is not None:
            raise ConnectionError("Есть открытые позиции! Сначала надо всех их закрыть :(")
            
    def handle_trade_message(self, message):
        # try:
        data = message.get('data')
        self.time = int(data[0].get("T"))#/60/int(cfg.period[1:]))
        time_rounded = int(int(data[0].get("T"))/1000/60/int(cfg.period[1:]))
        print ("\033[A\033[A")
        logger.info(f"server time: {datetime.fromtimestamp(int(self.time/1000))}")
        # except (ValueError, AttributeError):
            # pass            
        if time_rounded > self.t0:
            if self.t0:
                self.update()
                actpos = f"{self.open_position.ticker} {self.open_position.dir} {self.open_position.volume}" if self.open_position is not None else "пока нету"
                msg = f"{datetime.fromtimestamp(int(self.time/1000))}: processed new candle. Current pos: {actpos}"
                logger.info(msg)
                print()
                self.my_telebot.send_text(msg)
            self.t0 = time_rounded

        
    def trailing_sl(self, pos:Position):
        sl = float(pos.sl)
        # try:
        sl_new = float(sl + self.cfg.trailing_stop_rate*(self.h.Open[-1] - sl))
        sl = sl_new if abs(sl_new - self.h.Open[-1]) - self.cfg.ticksize > 0 else sl
        resp = self.session.set_trading_stop(
            category="linear",
            symbol=self.cfg.ticker,
            stopLoss=sl,
            slTriggerB="IndexPrice",
            positionIdx=0,
        )
        # logger.debug(resp)
        # except Exception as ex:
        #     print(ex)
        return float(sl)       

    def get_open_orders_positions(self):
            self.open_orders = []
            self.open_position = None
            self.open_orders = self.session.get_open_orders(category="linear", symbol=cfg.ticker)["result"]["list"]
            positions = self.session.get_positions(category="linear", symbol=cfg.ticker)["result"]["list"]
            for pos in positions :
                if float(pos["size"]):
                    self.open_position = Position(price=pos["avgPrice"],
                                                  date=pos["createdTime"],
                                                  indx=0,
                                                  ticker=pos["symbol"],
                                                  volume=pos["size"], 
                                                  period=self.cfg.period,
                                                  sl=pos["stopLoss"])

    def update(self):
        # try:
        self.get_open_orders_positions()    
        if cfg.save_plots:
            if self.hist2plot is not None:
                h = pd.DataFrame(self.h).iloc[-2:-1]
                h.index = pd.to_datetime(h.Date)
                self.hist2plot = pd.concat([self.hist2plot, h])
                self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
                log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                            
            if self.open_position is not None and self.hist2plot is None:
                self.hist2plot = pd.DataFrame(self.h)
                self.hist2plot.index = pd.to_datetime(self.hist2plot.Date)
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
                self.open_time = pd.to_datetime(self.hist2plot.iloc[-1].Date)
                self.side = self.open_position.str_dir
                self.lines2plot.append([(self.open_time, self.open_position.open_price), (self.open_time, self.open_position.open_price)])
                self.lines2plot.append([(self.open_time, self.open_position.sl), (self.open_time, self.open_position.sl)])
                log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, self.open_time, self.side, cfg.ticker))
                p.start()
                p.join()
                self.my_telebot.send_image(self.save_path / date2save_format(self.open_time))
                self.hist2plot = self.hist2plot.iloc[:-1]
                
                
        if self.open_position is not None:
            self.sl = self.trailing_sl(self.open_position)
            
        message = self.session.get_kline(
            category="linear",
            symbol=cfg.ticker,
            interval=cfg.period[1:],
            start=0,
            end=self.time,
            limit=cfg.hist_buffer_size
            )
        self.h = get_bybit_hist(message["result"], cfg.hist_buffer_size)
            
        if cfg.save_plots:
            if self.open_position is None and self.exp.order_sent and self.lines2plot: 
                last_row = pd.DataFrame(self.h).iloc[-2:]
                last_row.index = pd.to_datetime(last_row.Date)
                self.hist2plot = pd.concat([self.hist2plot, last_row])                    
                self.lines2plot[-2][-1] = (pd.to_datetime(self.hist2plot.iloc[-1].Date), self.lines2plot[-2][-1][-1])
                self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
                log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, self.open_time, self.side, cfg.ticker))
                p.start()
                p.join()
                self.my_telebot.send_image(self.save_path / date2save_format(self.open_time))
                self.hist2plot = None    
        
        texp = self.exp.update(self.h, self.open_position)
        # except Exception as ex:
        #     logger.error(ex)
        

    
if __name__ == "__main__":
    import sys
    import argparse
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to configuration file")
    parser.add_argument("--demo", action="store_true", help="use demo acc")
    args = parser.parse_args()

    cfg = PyConfig(args.config).test()
    demo = args.demo
    cfg.save_plots = True
    print(cfg)
    
    with open("./configs/api.yaml", "r") as f:
        creds = yaml.safe_load(f)
    if args.demo:
        creds["api_secret"] = creds["api_secret_demo"]
        creds["api_key"] = creds["api_key_demo"]
    
    public = WebSocket(channel_type='linear', testnet=False)
    # private = WebSocket(channel_type='private',
    #                     api_key=api_key,
    #                     api_secret=api_secret, 
    #                     testnet=False) 
    bybit_trading = BybitTrading(cfg, creds)
    bybit_trading.test_connection()
    public.trade_stream(symbol=cfg.ticker, callback=bybit_trading.handle_trade_message)
    
    print()
    while True:
        sleep(60)
        if not public.is_connected():
            logger.warning("Connection lost! Reconnect...")
            public.exit()
            public = WebSocket(channel_type='linear', testnet=False)
            public.trade_stream(symbol=cfg.ticker, callback=bybit_trading.handle_trade_message)
            sleep(1)
            msg = f"Request connection status: is_connected={public.is_connected()}\n"
            logger.warning(msg)
            bybit_trading.my_telebot.send_text(msg)
    

    
    