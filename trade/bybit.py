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
from trade.base import BaseTradeClass, StepData

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


class BybitTrading(BaseTradeClass):
    def __init__(self, cfg, expert, telebot, bybit_session) -> None:
        super().__init__(cfg=cfg, expert=expert, telebot=telebot)
        self.session = bybit_session
        self.nmin = int(self.cfg.period[1:])
        self.time = StepData()
            
    def handle_trade_message(self, message):
        time = self.to_datetime(message.get('data')[0].get("T"))
        time_rounded = time.astype("datetime64[m]").astype(int)//self.nmin
        self.time.update(np.datetime64(int(time_rounded*self.nmin), "m"))
        print ("\033[A\033[A")
        logger.info(f"server time: {date2str(time, 'ms')}: {date2str(self.time.prev)} -> {date2str(self.time.curr)}")
        
        if self.time.change(no_none=True):
            self.update()
            actpos = f"{self.pos.curr.ticker} {self.pos.curr.side} {self.pos.curr.volume}" if self.pos.curr is not None else "пусто"
            msg = f"{date2str(self.time.curr)}: cur. pos: {actpos}"
            logger.info(msg)
            print()
            self.my_telebot.send_text(msg)

    def update_trailing_stop(self, sl_new: float):
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

    def get_current_position(self):
            # self.open_orders = self.session.get_open_orders(category="linear", symbol=self.cfg.ticker)["result"]["list"]
            positions = self.session.get_positions(category="linear", symbol=self.cfg.ticker)["result"]["list"]
            positions = [pos for pos in positions if pos["size"] != "0"]
            return self._build_position(positions[0]) if len(positions) else None
                
    def _build_position(self, pos: Dict[str, Any]):
        return Position(
            price=float(pos["avgPrice"])*Side.from_str(pos["side"]).value,
            date=self.to_datetime(pos["updatedTime"]),
            indx=0,
            ticker=pos["symbol"],
            volume=float(pos["size"]),
            period=self.cfg.period,
            sl=float(pos["stopLoss"])
            )            

    def get_hist(self):
        message = self.session.get_kline(
            category="linear",
            symbol=self.cfg.ticker,
            interval=self.cfg.period[1:],
            start=0,
            end=self.time.curr,
            limit=self.cfg.hist_buffer_size
        )
        return get_bybit_hist(message["result"], self.cfg.hist_buffer_size)
        

def launch(cfg, demo=False):
    with open("./api.yaml", "r") as f:
        creds = yaml.safe_load(f)
    if demo:
        creds["api_secret"] = creds["api_secret_demo"]
        creds["api_key"] = creds["api_key_demo"]
    
    bybit_session = HTTP(testnet=False, api_key=creds["api_key"], api_secret=creds["api_secret"])
    bybit_trading = BybitTrading(cfg=cfg, 
                                 expert=ByBitExpert(cfg, bybit_session), 
                                 telebot=Telebot(creds["bot_token"]), 
                                 bybit_session=bybit_session)
    bybit_trading.test_connection()    
    public = WebSocket(channel_type='linear', testnet=False)
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
    

    
    