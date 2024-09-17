from typing import Any, List, Optional

import numpy as np
from loguru import logger

from common.type import Side
from common.utils import FeeConst, FeeModel, date2str


class Order:
    class TYPE:
        MARKET = "market order"
        LIMIT = "limit order"

    def __init__(self, directed_price, type, volume, indx, date):
        self.side = Side.from_int(np.sign(directed_price))
        self.type = type
        self.volume = volume
        self.open_indx = indx
        self.open_date = date
        self.close_date = None
        self._change_hist = []
        self.change(date, abs(directed_price) if type == Order.TYPE.LIMIT else 0)

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

    def change(self, date, price):
        assert round(date) == date
        self.price = price
        if price != 0:
            self._change_hist.append((date, price))

    def close(self, date):
        assert round(date) == date
        self.close_date = date
        self._change_hist.append((date, self.price))


class Position:
    def __init__(
        self,
        price: float,
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
        self.open_price = abs(float(price))
        self.open_date = np.datetime64(date)
        self.open_indx = int(indx)
        self.sl = float(sl) if sl is not None else sl
        self.open_risk = np.nan
        if self.sl is not None:
            self.open_risk = abs(self.open_price - self.sl) / self.open_price * 100
        self.side: Side = Side.from_int(np.sign(float(price)))
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
        return f"pos {self.ticker} {self.side} {self.volume} {self.id}"

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

    @property
    def lines(self):
        return [(self.open_indx, self.open_price), (self.close_indx, self.close_price)]
