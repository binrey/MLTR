from enum import Enum
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from common.type import Side
from common.utils import FeeConst, FeeModel, date2str


class ORDER_TYPE(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOPLOSS = "stoploss"
    
class Order:
    def __init__(self, price: float, side: Side, type: ORDER_TYPE, volume: int, indx: int, time: np.datetime64):
        self.price = price
        self.side = side
        self.type = type
        self.volume = volume
        self.open_indx = indx
        self.open_date = time
        self.close_date = None

    def __str__(self):
        return f"{self.type} {self.id} {self.volume}"

    @property
    def str_dir(self):
        return "BUY" if self.side.value > 0 else "SELL"

    @property
    def id(self):
        return f"{self.open_indx}-{self.str_dir}-{self.price:.2f}"

    @property
    def lines(self):
        return self._change_hist

    def close(self, date):
        self.close_date = date


class Position:
    def __init__(
        self,
        price: float,
        side: Side,
        date: Any,
        indx: int,
        ticker: str = "NoName",
        volume: int = 1,
        period: str = "M5",
        sl: Optional[float] = None,
        fee_rate: Optional[FeeModel] = None,
    ):
        self.volume = float(volume)
        assert self.volume > 0
        self.ticker = ticker
        self.period = period
        self.open_price = float(price)
        self.side: Side = side
        self.open_date = np.datetime64(date)
        self.open_indx = int(indx)
        self.sl = None
        self.sl_hist = []
        
        self.update_sl(sl=float(sl) if sl is not None else sl, time=self.open_date)
        self.close_price = None
        self.close_date = None
        self.profit = None
        self.profit_abs = None
        self.fee_rate = fee_rate if fee_rate is not None else FeeConst(0, 0)
        self.fees = 0
        self.fees_abs = 0
        self._update_fees(self.open_price, self.volume)
        logger.debug(f"{date2str(date)} open position {self.id}")

    def __str__(self):
        return f"pos {self.ticker} {self.side} {self.volume}: {self.open_price}"

    def update_sl(self, sl: float, time: np.datetime64):
        assert not (self.sl is not None and sl is None), "Set sl to None is not allowed"
        logger.debug(f"{date2str(time)} update sl {self.sl} -> {sl}")
        self.sl = sl
        if sl is not None:
            self.sl_hist.append((time, sl))

    def _update_fees(self, price, volume):
        self.fees_abs += self.fee_rate.order_execution_fee(price, volume)
        if self.close_date and self.open_date:
            self.fees_abs += self.fee_rate.position_suply_fee(
                self.open_date,
                self.close_date,
                (self.open_price + self.close_price) / 2,
                volume,
            )
        self.fees = self.fees_abs / self.volume / self.open_price * 100

    @property
    def str_dir(self):
        return "BUY" if self.side.value > 0 else "SELL"

    @property
    def duration(self):
        return self.close_indx - self.open_indx

    @property
    def id(self):
        return (
            f"{self.open_indx}-{self.str_dir}-{self.open_price:.2f}-{self.volume:.2f}"
        )

    def close(self, price, date, indx):
        self.close_price = abs(price)
        self.close_date = np.datetime64(date)
        self.close_indx = indx
        self._update_fees(self.close_price, self.volume)
        self.profit_abs = (self.close_price - self.open_price) * self.side.value * self.volume
        self.profit_abs -= self.fees_abs
        self.profit = self.profit_abs / self.open_price * 100
        logger.debug(
            f"{date2str(date)} close position {self.id} at {self.close_price:.2f}, profit: {self.profit_abs:.2f} ({self.profit:.2f}%)"
        )

    def cur_profit(self, price):
        return (price - self.open_price) * self.side.value * self.volume

    @property
    def lines(self):
        return [(self.open_indx, self.open_price), (self.close_indx, self.close_price)]


def fix_rate_trailing_sl(sl:float, 
                       open_price: float, 
                       side: Side,
                       trailing_stop_rate:float, 
                       ticksize: float) -> float:
    sl_new = float(sl + trailing_stop_rate*(open_price - sl))
    return sl_new