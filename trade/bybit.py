from time import sleep

import pandas as pd
from loguru import logger
from tqdm import tqdm

from common.type import Side
from common.utils import Telebot
from data_processing.dataloading import DTYPE
from trade.utils import Position

pd.options.mode.chained_assignment = None
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import stackprinter
import yaml
from pybit.unified_trading import HTTP, WebSocket

from experts.experts import ByBitExpert
from trade.base import BaseTradeClass, log_get_hist

stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen
 
def get_bybit_hist(mresult, size):
    data = np.zeros(size, dtype=DTYPE)
    
    input = np.array(mresult["list"], dtype=np.float64)[::-1]
    data['Id'] = input[:, 0].astype(np.int64)
    data['Date'] = data['Id'].astype("datetime64[ms]")
    data['Open'] = input[:, 1]
    data['High'] = input[:, 2]
    data['Low'] = input[:, 3]
    data['Close'] = input[:, 4]
    data['Volume'] = input[:, 5].astype(np.int64)
    
    return data


class BybitTrading(BaseTradeClass):
    def __init__(self, cfg, telebot: Telebot, bybit_session: HTTP) -> None:
        self.session = bybit_session
        super().__init__(cfg=cfg, expert=ByBitExpert(cfg, bybit_session), telebot=telebot)

    def get_server_time(self) -> np.datetime64:
        serv_time = int(self.session.get_server_time()["result"]["timeSecond"])
        serv_time = np.array(serv_time).astype("datetime64[s]")
        return serv_time     
       
    def to_datetime(self, timestamp: Union[int, float]) -> np.datetime64:
        return np.datetime64(int(timestamp), "ms")
    
    def get_pos_history(self, limit=5):
        positions = self.session.get_closed_pnl(category="linear", limit=limit)["result"]["list"]
        positions = [self._build_position(pos) for pos in positions if pos["symbol"] == self.ticker]
        return positions

    def get_current_position(self):
            positions = self.session.get_positions(category="linear", symbol=self.ticker)["result"]["list"]
            positions = [pos for pos in positions if pos["size"] != "0"]
            return self._build_position(positions[0]) if len(positions) else None
                
    def get_qty_step(self):
        msg = self.session.get_instruments_info(
            category="linear",
            symbol=self.ticker,
        )["result"]
        return float(msg["list"][0]["lotSizeFilter"]["qtyStep"])
                
    def _build_position(self, pos: Dict[str, Any]):
        if "avgEntryPrice" in pos.keys():
            price = pos["avgEntryPrice"]
        else:
            price = pos.get("avgPrice", 0)

        pos_object = Position(
            price=float(price),
            date=self.to_datetime(pos["updatedTime"]),
            indx=0,
            side=Side.from_str(pos["side"]),
            ticker=pos["symbol"],
            volume=float(pos["closedSize"] if "closedSize" in pos else pos["size"]),
            period=self.cfg.period,
            sl=float(pos["stopLoss"]) if "stopLoss" in pos and len(pos["stopLoss"]) else None,
            )    
        if "avgExitPrice" in pos.keys():
            pos_object.close(price=float(pos["avgExitPrice"]),
                      date=self.to_datetime(pos["updatedTime"]),
                      indx=int(pos["updatedTime"])
                      )
        return pos_object

    @log_get_hist
    def get_hist(self):
        t = self.time.curr
        data = None
        while data is None or t != data["Date"][-1]:
            logger.info(f"request history data for {t}...")
            message = self.session.get_kline(
                category="linear",
                symbol=self.ticker,
                interval=str(self.cfg.period.minutes),
                start=0,
                end=t.astype("datetime64[ms]").astype(int),
                limit=self.cfg.hist_buffer_size
            )
            data = get_bybit_hist(message["result"], self.cfg.hist_buffer_size)
        return data
    

def launch(cfg, demo=False):
    with open("./api.yaml", "r") as f:
        creds = yaml.safe_load(f)
    if demo:
        creds["api_secret"] = creds["api_secret_demo"]
        creds["api_key"] = creds["api_key_demo"]
    
    bybit_session = HTTP(testnet=False, api_key=creds["api_key"], api_secret=creds["api_secret"])
    bybit_trading = BybitTrading(cfg=cfg,
                                 telebot=Telebot(creds["bot_token"]), 
                                 bybit_session=bybit_session)
    bybit_trading.test_connection()
    
    print()
    while True:
        bybit_trading.handle_trade_message(None)
        time_step_curr = bybit_trading.time.curr
        time_step_next = time_step_curr + np.timedelta64(cfg.period.minutes, "m")
        dtime = time_step_next - bybit_trading.get_server_time()
        # for _ in tqdm(range(dtime.astype(int)), "wait"):
        #     sleep(1)
        print(f"wait {dtime} ...")
        sleep(dtime.astype(int))
        # print ("\033[A\033[A")
    