from enum import Enum
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from common.type import Line, Point, Side
from common.utils import FeeConst, FeeModel, date2str


class ORDER_TYPE(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOPLOSS = "stoploss"
    TAKEPROF = "takeprofit"


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
        tp: Optional[float] = None,
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
        self.tp = None
        self.sl_hist = []
        self.tp_hist = []
        self.enter_points_hist: List[Point] = []
        self.enter_price_hist: List[Point] = []
        self.volume_hist: List[Point] = []

        # Record the initial stop-loss and take-profit if provided.
        self.update_sl(sl=float(sl) if sl is not None else sl, time=self.open_date)
        self.update_tp(tp=float(tp) if tp is not None else tp, time=self.open_date)

        self.close_price = None
        self.close_date = None
        self.profit = None
        self.profit_abs = None
        self.fee_rate = fee_rate if fee_rate is not None else FeeConst(0, 0)
        self.fees = 0
        self.fees_abs = 0
        self._update_fees(self.open_price, self.volume)

        # Record price change events. Each tuple is (date, price)
        
        self.enter_points_hist.append((pd.to_datetime(self.open_date), self.open_price))
        self.enter_price_hist.append((pd.to_datetime(self.open_date), self.open_price))
        self.volume_hist.append((pd.to_datetime(self.open_date), self.volume))

    def __str__(self):
        name = f"pos {self.ticker} {self.side} {self.volume}: {self.open_price}"
        if self.close_price is not None:
            name += f" -> {self.close_price}"
        if self.sl is not None:
            name += f" | sl:{self.sl}"
        if self.tp is not None:
            name += f" | tp:{self.tp}"
        return name

    def update_sl(self, sl: float, time: np.datetime64):
        assert not (self.sl is not None and sl is None), "Set sl to None is not allowed"
        self.sl = sl
        if sl is not None:
            self.sl_hist.append((time, sl))

    def update_tp(self, tp: float, time: np.datetime64):
        assert not (self.tp is not None and tp is None), "Set tp to None is not allowed"
        self.tp = tp
        if tp is not None:
            self.tp_hist.append((time, tp))

    def _update_fees(self, price, volume):
        self.fees_abs += self.fee_rate.order_execution_fee(price, volume)
        if self.close_date is not None and self.open_date is not None:
            self.fees_abs += self.fee_rate.position_suply_fee(
                self.open_date,
                self.close_date,
                (self.open_price + self.close_price) / 2,
                volume,
            )
        self.fees = self.fees_abs / self.volume / self.open_price * 100

    def _update_profit_abs(self, price, volume):
        if self.profit_abs is None:
            self.profit_abs = 0
        self.profit_abs += (price - self.open_price) * self.side.value * volume        

    @property
    def str_dir(self):
        return "BUY" if self.side.value > 0 else "SELL"

    @property
    def duration(self):
        return self.close_indx - self.open_indx

    @property
    def id(self):
        return f"{self.open_indx}-{self.str_dir}-{self.open_price:.2f}-{self.volume:.2f} sl:{self.sl}"

    def close(self, price, date, indx):
        self.close_price = abs(price)
        self.close_date = np.datetime64(date)
        self.close_indx = indx
        self._update_fees(self.close_price, self.volume)
        self._update_profit_abs(self.close_price, self.volume)
        self.profit = (self.profit_abs - self.fees_abs) / self.open_price * 100
        self.enter_points_hist.append((self.close_date, self.close_price))
        self.enter_price_hist.append((self.close_date, self.open_price))

    def cur_profit(self, price):
        return (price - self.open_price) * self.side.value * self.volume

    @property
    def lines(self):
        """
        Return all (date, price) pairs that represent changes in the position's effective price.
        This includes the open event, any add-to-position events, and the closing event.
        """
        return self.enter_points_hist

    def add_to_position(self, 
                        additional_volume: int, 
                        price: float, 
                        time: np.datetime64):
        """
        Add to the existing position by increasing the volume and updating fees.
        Also record the event as a price change.
        
        :param additional_volume: The volume to add to the current position.
        :param price: The price at which the additional volume is added.
        :param time: The timestamp for this volume change. If not provided, the current time is used.
        """
        if additional_volume <= 0:
            return
        self.open_price = self.open_price * self.volume + price * additional_volume
        self.volume += additional_volume
        self.open_price /= self.volume
        self._update_fees(price, additional_volume)
        
        self.enter_points_hist.append((pd.to_datetime(time), price))
        self.enter_price_hist.append((pd.to_datetime(time), self.open_price))
        self.volume_hist.append((pd.to_datetime(self.open_date), self.volume))
        logger.debug(f"Added {additional_volume} to position {self.id} at price {price}")

    def trim_position(self, 
                      trim_volume: int, 
                      price: float, 
                      time: np.datetime64):
        """
        Trim existing position by decreasing the volume and updating fees and partial profit.
        
        Parameters:
            trim_volume (int): The volume to trim (i.e. exit) from the current position.
            price (float): The execution price for the trimmed volume.
            time (np.datetime64): The timestamp of the trim event.
        """
        assert trim_volume < self.volume, "Cannot trim more or equal than current position"

        self._update_profit_abs(price, trim_volume)
        self._update_fees(price, trim_volume)
        self.volume -= trim_volume
        self.enter_points_hist.append((pd.to_datetime(time), price))
        self.enter_price_hist.append((pd.to_datetime(time), self.open_price))
        self.volume_hist.append((pd.to_datetime(time), self.volume))
        logger.debug(f"Remove {trim_volume} from position {self.id} at price {price}")
        
    def get_drawitem(self):
        return {"enter_points": Line(points=self.enter_points_hist, color='#8c8' if self.side == Side.BUY else '#c88'),
                "enter_price": Line(points=self.enter_price_hist),
                "volume": Line(points=self.volume_hist)}
        