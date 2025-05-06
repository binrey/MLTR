from time import sleep

import pandas as pd
from loguru import logger

from common.type import Side
from common.utils import Telebot
from data_processing.dataloading import DTYPE
from trade.utils import Position

pd.options.mode.chained_assignment = None
from typing import Any, Dict, Optional, Union

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
        positions = self.session.get_closed_pnl(category="linear",
                                                symbol=self.ticker,
                                                limit=limit)["result"]["list"]
        return positions

    def _get_pos_from_message_dict(self, pos_dict):
        ticker, price, volume, sl, side, date = self._parse_bybit_position(pos_dict)
        pos = Position(
            price=float(price),
            date=date,
            indx=0,
            side=side,
            ticker=ticker,
            volume=volume,
            period=self.period,
            sl=sl,
        )
        logger.debug(f"Getting pos from bybit: {pos}...")
        return pos
        

    def _close_current_pos(self):
        pos_dict = self.get_pos_history(1)[0]
        self.pos.curr.close(
            price=float(pos_dict["avgExitPrice"]),
            date=self.to_datetime(pos_dict["updatedTime"]),
            indx=int(pos_dict["updatedTime"])
            )
        logger.debug(f"Closing self.pos.curr: {self.pos.curr}")

    def get_current_position(self):
        pos_object: Optional[Position] = None
        dict2position = None
        open_positions = self.session.get_positions(category="linear", symbol=self.ticker)["result"]["list"]
        open_positions = [pos for pos in open_positions if pos["size"] != "0"]
        if self.pos.curr:
            if len(open_positions) == 0:
                self._close_current_pos()
            elif Side.from_str(open_positions[0]["side"]) != self.pos.curr.side:
                self._close_current_pos()
                dict2position = open_positions[0]
            else:
                pos_dict = open_positions[0]
                _, _, _, sl, _, _ = self._parse_bybit_position(pos_dict)
                pos_object = self.pos.curr
                if sl is not None:
                    pos_object.update_sl(sl, self.time.prev)
                # TODO: add to position
                # pos_object.add_to_position()
                # TODO: update tp
                # pos_object.update_tp(pos["takeProfit"], self.time.prev)
                return pos_object

        elif len(open_positions):
            dict2position = open_positions[0]

        if dict2position is not None:
            pos_object = self._get_pos_from_message_dict(dict2position)
        return pos_object
    
    def get_qty_step(self):
        msg = self.session.get_instruments_info(
            category="linear",
            symbol=self.ticker,
        )["result"]
        return float(msg["list"][0]["lotSizeFilter"]["qtyStep"])

    def _parse_bybit_position(self, pos: Dict[str, Any]):
        sl = float(pos["stopLoss"]) if "stopLoss" in pos and len(pos["stopLoss"]) else None
        volume = float(pos["closedSize"] if "closedSize" in pos else pos["size"])
        date = self.to_datetime(pos["updatedTime"])
        side = Side.from_str(pos["side"])
        ticker = pos["symbol"]
        price = pos["avgEntryPrice"] if "avgEntryPrice" in pos else pos.get("avgPrice", 0)
        return ticker, price, volume, sl, side, date

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
        total_seconds = self.period.to_timedelta().astype(int)*60
        remaining_seconds = (next_update_time - self.get_server_time()).astype(int)
        while remaining_seconds > 0:
            minutes, seconds = divmod(remaining_seconds, 60)
            progress = int(((total_seconds - remaining_seconds) / total_seconds) * 30) if total_seconds > 0 else 0
            bar = '[' + '#' * progress + '-' * (30 - progress) + ']'
            print(f"\rWaiting {minutes:02d}m {seconds:02d}s {bar}", end="", flush=True)
            sleep_time = min(1, remaining_seconds)
            sleep(sleep_time)
            remaining_seconds -= sleep_time
        print("\rWaiting 00m 00s [##############################]", flush=True)  # Clear line at the end


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
    