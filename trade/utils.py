import json
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from common.type import Line, Point, Side, to_datetime
from common.utils import FeeConst, FeeModel, date2str
from data_processing.dataloading import DTYPE


def log_creating_order(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, side: Side, volume: float, time_id: Optional[int] = None):
        logger.debug(f"Creating order {side} {volume}...")
        result = func(self, side, volume, time_id)
        if isinstance(result, list):
            result = " ".join(map(str, result))
        logger.debug(f"Orders created: {result}")
        return result
    return wrapper


def log_modify_sl(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, sl: Optional[float]):
        current_pos = self.get_current_position()
        if current_pos is not None:
            logger.debug(f"Modifying sl: {current_pos.sl} -> {sl}")
        result = func(self, sl)
        return result
    return wrapper


def log_modify_tp(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, tp: Optional[float]):
        current_pos = self.get_current_position()
        if current_pos is not None:
            logger.debug(f"Modifying tp: {current_pos.tp} -> {tp}")
        result = func(self, tp)
        return result
    return wrapper


def get_bybit_hist(mresult):
    input = np.array(mresult["list"], dtype=np.float64)[::-1]
    data = np.zeros(input.shape[0], dtype=DTYPE)

    data['Id'] = input[:, 0].astype(np.int64)
    data['Date'] = data['Id'].astype("datetime64[ms]")
    data['Open'] = input[:, 1]
    data['High'] = input[:, 2]
    data['Low'] = input[:, 3]
    data['Close'] = input[:, 4]
    data['Volume'] = input[:, 5]

    return data


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
        qty_step: float = 1,
        period: str = "M5",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        fee_rate: Optional[FeeModel] = None,
        fee: float = None,
    ):
        self.vol_round = int(1/qty_step)
        self.volume: Fraction = self.set_volume(volume)
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
        if sl is not None:
            self.update_sl(sl=float(sl), time=self.open_date)
        if tp is not None:
            self.update_tp(tp=float(tp), time=self.open_date)

        self.close_price = None
        self.close_date = None
        self.close_indx = None
        self.profit = None
        self.profit_abs = None
        self.fee_rate = fee_rate if fee_rate is not None else FeeConst(0, 0)
        self.fees = 0
        self.fees_abs = 0
        self._update_fees(self.open_price, self.volume, fee)
        
        self.enter_points_hist.append((to_datetime(self.open_date), self.open_price))
        self.enter_price_hist.append((to_datetime(self.open_date), self.open_price))
        self.volume_hist.append((to_datetime(self.open_date), float(self.volume)))

    def __str__(self):
        name = f"pos {self.ticker} {self.side} opened.{date2str(self.open_date, 'm')} vol.{float(self.volume)} p.{self.open_price}"
        if self.close_price is not None:
            name += f" -> {self.close_price}"
        if self.sl is not None:
            name += f" sl.{self.sl}"
        if self.tp is not None:
            name += f" tp.{self.tp}"
        name += f" | cost.{self.cost:.2f}"
        return name

    def set_volume(self, volume: float) -> Fraction:
        return Fraction(int(round(volume*self.vol_round, 0)), self.vol_round)
    
    @property
    def cost(self):
        return self.open_price * self.volume

    def update_sl(self, sl: float, time: np.datetime64):
        assert not (self.sl is not None and sl is None), "Set sl to None is not allowed"
        self.sl = sl
        if not any(t == time for t, _ in self.sl_hist):
            self.sl_hist.append((time, sl))

    def update_tp(self, tp: float, time: np.datetime64):
        assert not (self.tp is not None and tp is None), "Set tp to None is not allowed"
        self.tp = tp
        self.tp_hist.append((time, tp))

    def _update_fees(self, price, volume, fee: Optional[float] = None):
        if fee is None:
            fee = self.fee_rate.order_execution_fee(price, volume)
            if self.close_date is not None and self.open_date is not None:
                fee += self.fee_rate.position_suply_fee(
                    self.open_date,
                    self.close_date,
                    (self.open_price + self.close_price) / 2,
                    volume,
                )
        self.fees_abs += fee
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
        return f"{self.open_indx}-{self.str_dir}-{self.open_price:.2f}-{float(self.volume.real):.2f} sl:{self.sl}"

    def close(self, price, date, indx, fee: Optional[float] = None):
        self.close_price = abs(price)
        self.close_date = np.datetime64(date)
        self.close_indx = indx
        self._update_fees(self.close_price, self.volume, fee)
        self._update_profit_abs(self.close_price, self.volume)
        self.profit = (self.profit_abs - self.fees_abs) / self.open_price * 100
        self.enter_points_hist.append((to_datetime(self.close_date), self.close_price))
        self.enter_price_hist.append((to_datetime(self.close_date), self.open_price))

    def unrealized_pnl(self, price):
        return (price - self.open_price) * self.side.value * self.volume

    @property
    def lines(self):
        """
        Return all (date, price) pairs that represent changes in the position's effective price.
        This includes the open event, any add-to-position events, and the closing event.
        """
        return self.enter_points_hist

    def add_to_position(self, 
                        additional_volume: float, 
                        price: float, 
                        time: np.datetime64,
                        fee: Optional[float] = None):
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
        self.volume += self.set_volume(additional_volume)
        self.open_price /= self.volume
        self._update_fees(price, additional_volume, fee)
        
        self.enter_points_hist.append((to_datetime(time), price))
        self.enter_price_hist.append((to_datetime(time), self.open_price))
        self.volume_hist.append((to_datetime(time), float(self.volume)))
        logger.debug(f"Added {additional_volume} to position {self.id} at price {price}")

    def trim_position(self,
                      trim_volume: int,
                      price: float,
                      time: np.datetime64,
                      fee: Optional[float] = None):
        """
        Trim existing position by decreasing the volume and updating fees and partial profit.
        
        Parameters:
            trim_volume (int): The volume to trim (i.e. exit) from the current position.
            price (float): The execution price for the trimmed volume.
            time (np.datetime64): The timestamp of the trim event.
        """
        trim_volume = self.set_volume(trim_volume)
        assert trim_volume < self.volume, "Cannot trim more or equal than current position"

        self._update_profit_abs(price, trim_volume)
        self._update_fees(price, trim_volume, fee)
        self.volume -= trim_volume
        self.enter_points_hist.append((to_datetime(time), price))
        self.enter_price_hist.append((to_datetime(time), self.open_price))
        self.volume_hist.append((to_datetime(time), float(self.volume)))
        logger.debug(f"Remove {trim_volume} from position {self.id} at price {price}")
        
    def get_drawitem(self):
        return {"enter_points": Line(points=self.enter_points_hist, color='#8c8' if self.side == Side.BUY else '#c88'),
                "enter_price": Line(points=self.enter_price_hist),
                "volume": Line(points=self.volume_hist)}

    def to_dict(self) -> dict:
        """Convert position data to a dictionary that can be safely serialized to YAML.
        
        Returns:
            dict: Dictionary containing all relevant position data in a YAML-serializable format
        """
        def convert_datetime(dt):
            if dt is None:
                return None
            return pd.Timestamp(dt).isoformat()

        return {
            "ticker": self.ticker,
            "side": str(self.side),
            "volume": float(self.volume),
            "open_price": float(self.open_price),
            "close_price": float(self.close_price) if self.close_price is not None else None,
            "fees_abs": float(self.fees_abs),
            "stop_loss": float(self.sl) if self.sl is not None else None,
            "take_profit": float(self.tp) if self.tp is not None else None,
            "open_date": convert_datetime(self.open_date),
            "open_indx": int(self.open_indx),
            "close_date": convert_datetime(self.close_date),
            "close_indx": int(self.close_indx),
            "period": str(self.period),
            "id": self.id,
            "sl_history": [(convert_datetime(t), float(p)) for t, p in self.sl_hist],
            "tp_history": [(convert_datetime(t), float(p)) for t, p in self.tp_hist],
            "enter_points": [(convert_datetime(t), float(p)) for t, p in self.enter_points_hist],
            "enter_prices": [(convert_datetime(t), float(p)) for t, p in self.enter_price_hist],
            "volume_history": [(convert_datetime(t), float(v)) for t, v in self.volume_hist]
        }
        
    @staticmethod
    def from_dict(data: dict) -> "Position":
        """Create a Position instance from a dictionary.
        
        Args:
            data (dict): Dictionary containing position data
            
        Returns:
            Position: New Position instance
        """
        # Convert string dates back to numpy datetime64
        open_date = np.datetime64(data['open_date']) if data['open_date'] else None
        open_indx = int(data['open_indx']) if 'open_indx' in data else 0
        close_date = np.datetime64(data['close_date']) if data['close_date'] else None
        close_indx = int(data['close_indx']) if 'close_indx' in data else 0
        
        # Create the position
        position = Position(
            price=data['open_price'],
            side=Side[data['side']],
            date=open_date,
            indx=open_indx,
            ticker=data['ticker'],
            volume=data['volume'],
            qty_step=0.001,#data['qty_step'],
            period=data['period'],
            sl=data['stop_loss'],
            tp=data['take_profit'],
            fee_rate=None,
            fee=data['fees_abs']
        )
        
        # Set additional attributes if they exist
        if close_date is not None and data['close_price'] is not None:
            position.close(data['close_price'], close_date, close_indx)
        
        # Restore history arrays
        position.sl_hist = [(np.datetime64(t), float(p)) for t, p in data['sl_history']]
        position.tp_hist = [(np.datetime64(t), float(p)) for t, p in data['tp_history']]
        position.enter_points_hist = [(np.datetime64(t), float(p)) for t, p in data['enter_points']]
        position.enter_price_hist = [(np.datetime64(t), float(p)) for t, p in data['enter_prices']]
        position.volume_hist = [(np.datetime64(t), float(v)) for t, v in data['volume_history']]
        
        return position
    
    @staticmethod
    def from_json_file(file_path: str) -> "Position":
        return Position.from_dict(json.load(open(file_path, "r")))