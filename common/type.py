import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd


class ConfigType(Enum):
    BACKTEST = "backtest"
    OPTIMIZE = "optimization"
    CROSS_VALIDATION = "cross_validation"
    BYBIT = "bybit"
    MULTIRUN = "multirun"

class Side(Enum):
    BUY = 1
    SELL = -1
    NONE = 0

    @staticmethod
    def from_str(side: str, none_if_invalid=True):
        if side.lower() == "buy":
            return Side.BUY
        elif side.lower() == "sell":
            return Side.SELL
        else:
            if none_if_invalid:
                return None
            else:
                raise ValueError(f"{side} is not valid value, set ['buy' or 'sell']")

    @staticmethod
    def from_int(side: int):
        if side > 0:
            return Side.BUY
        elif side < 0:
            return Side.SELL
        else:
            return Side.NONE

    def __str__(self):
        return self.name
    
    @staticmethod
    def reverse(side):
        return Side.BUY if side == Side.SELL else Side.SELL


# Type aliases for stop-loss and take-profit price definitions
SLDefiner = dict[Side, float | None]
TPDefiner = dict[Side, float | None]

def to_datetime(time: np.datetime64):
    return pd.to_datetime(time, format="%Y-%m-%d", cache=True)


class Vis(Enum):
    ON_STEP = 0
    ON_DEAL = 1
    

class TimePeriod(Enum):
    M60 = "M60"
    M15 = "M15"
    M5 = "M5"
    M1 = "M1"

    def to_timedelta(self):
        return np.timedelta64(self.value[1:], self.value[0].lower())
    
    @property
    def minutes(self):
        return int(self.value[1:])
    
    @property
    def hours(self):
        return int(self.value[1:])/60
    
    def to_days(self, value: float):
        return value/self.hours/24

class RunType(Enum):
    BACKTEST = "backtest"
    OPTIMIZE = "optimization"
    CROSS_VALIDATION = "cross_validation"
    BYBIT = "bybit"
    MULTIRUN = "multirun"

    @classmethod
    def from_str(cls, label):
        if label.upper() in cls.__members__:
            return cls[label.upper()]
        else:
            raise ValueError(f"Unknown run type: {label}")

@dataclass
class VolEstimRule(Enum):
    FIXED_POS_COST = "fixed_pos_cost"
    DEPOSIT_BASED = "deposit_based"

@dataclass
class VolumeControl:
    rule: VolEstimRule
    deposit_fraction: float = 1

    def define(self, deposit: float) -> float:
        return deposit*self.deposit_fraction if self.rule == VolEstimRule.DEPOSIT_BASED else deposit

Point = Tuple[np.datetime64, float]
Bar = Tuple[float, float]

@dataclass
class Line:
    """Represents a line with points and color attributes.
    
    Attributes:
        points (List[Point]): List of (timestamp, value) points defining the line
        color (str): Color of the line for visualization
    """
    points: List[Point] = field(default_factory=list)
    color: str = "black"
    width: int = 4

    def to_dataframe(self):
        df = pd.DataFrame(self.points, columns=["Date", "Close"])
        df = df.dropna()
        
        df["Date"] = to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df


@dataclass
class TimeVolumeProfile:
    time: np.datetime64
    hist: List[Bar]

    def __post_init__(self):
        if isinstance(self.time, np.datetime64):
            self.time = self.time.astype("datetime64[m]")

    def to_datetime(self) -> None:
        self.time = to_datetime(self.time)


@dataclass
class Symbol:
    ticker: str = None
    tick_size: float = None
    qty_step: float = None
    stops_step: float = None

    @classmethod
    def qty_digits(cls, qty_step: float):
        return len(str(qty_step).split(".")[1])

    @classmethod
    def round_qty(cls, qty: float, qty_step: float):
        return round(math.floor(qty / qty_step) * qty_step, cls.qty_digits(qty_step))

    @classmethod
    def round_price(cls, tick_size: float, price: float):
        return round(math.floor(price / tick_size) * tick_size, cls.qty_digits(tick_size)) # TODO: check if this is correct

    @classmethod
    def round_stops(cls, stops_step: float, stops: float):
        return round(math.floor(stops / stops_step) * stops_step, cls.qty_digits(stops_step))
    
class Symbols:
    BTCUSDT = Symbol(ticker="BTCUSDT", tick_size=0.01, qty_step=0.001, stops_step=0.1)
    ETHUSDT = Symbol(ticker="ETHUSDT", tick_size=0.01, qty_step=0.01, stops_step=0.01)
    XRPUSDT = Symbol(ticker="XRPUSDT", tick_size=0.0001, qty_step=1, stops_step=0.1)
    SOLUSDT = Symbol(ticker="SOLUSDT", tick_size=0.001, qty_step=0.1, stops_step=0.1)
