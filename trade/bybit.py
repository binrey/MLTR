from time import sleep

import pandas as pd
from loguru import logger

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
from pybit.unified_trading import HTTP

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
        logger.info(f"Initialized BybitTrading with ticker: {self.ticker}, period: {self.period}")

    def get_server_time(self) -> np.datetime64:
        while True:
            try:
                serv_time = int(self.session.get_server_time()["result"]["timeSecond"])
                return np.datetime64(serv_time, "[s]")
            except Exception as e:
                logger.error(f"Error getting server time: {e}")
                sleep(1)
       
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
            period=self.period,
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
            logger.debug(f"Requesting history data for {t}...")
            message = self.session.get_kline(
                category="linear",
                symbol=self.ticker,
                interval=str(self.period.minutes),
                start=0,
                end=t.astype("datetime64[ms]").astype(int),
                limit=self.hist_size
            )
            data = get_bybit_hist(message["result"], self.hist_size)
        return data
    
    def wait_until_next_update(self, next_update_time):
        remaining_seconds = (next_update_time - self.get_server_time()).astype(int)
        while remaining_seconds > 0:
            minutes, seconds = divmod(remaining_seconds, 60)
            logger.debug(f"Waiting {minutes}m {seconds}s until next update...")
            sleep_time = min(10, remaining_seconds)
            sleep(sleep_time)
            remaining_seconds -= sleep_time


def launch(cfg, demo=False):
    with open("./api.yaml", "r") as f:
        api = yaml.safe_load(f)
    bybit_creds = api["bybit_demo"] if demo else api[cfg["credentials"]]
    bot_token = api["bot_token"]
    
    bybit_session = HTTP(testnet=False,
                         api_key=bybit_creds["api_key"],
                         api_secret=bybit_creds["api_secret"],
                         demo=demo)
    logger.info(f"Starting Bybit trading session (demo={demo})")
    bybit_trading = BybitTrading(cfg=cfg,
                                 telebot=Telebot(bot_token), 
                                 bybit_session=bybit_session)
    bybit_trading.initialize()
    
    print()
    while True:
        bybit_trading.handle_trade_message(None)
        time_step_curr = bybit_trading.time.curr
        time_step_next = time_step_curr + np.timedelta64(bybit_trading.period.minutes, "m")
        bybit_trading.wait_until_next_update(time_step_next)
    