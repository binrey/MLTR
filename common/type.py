from enum import Enum

import numpy as np


class Side(Enum):
    BUY = 1
    SELL = -1

    @staticmethod
    def from_str(side: str):
        if side.lower() == "buy":
            return Side.BUY
        elif side.lower() == "sell":
            return Side.SELL
        else:
            raise ValueError(f"{side} is not valid value, set ['buy' or 'sell']")

    @staticmethod
    def from_int(side: int):
        if side > 0:
            return Side.BUY
        elif side < 0:
            return Side.SELL
        else:
            raise ValueError(f"{side} undefined")

    def __str__(self):
        return "BUY" if self == Side.BUY else "SELL"
    
    @staticmethod
    def reverse(side):
        return Side.BUY if side == Side.SELL else Side.SELL


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
        
class RunType(Enum):
    BACKTEST = "backtest"
    OPTIMIZE = "optimize"
    BYBIT = "bybit"

    @staticmethod
    def from_str(label):
        if label in ('backtest', 'optimize', 'bybit'):
            return RunType[label.upper()]
        else:
            raise ValueError(f"Unknown run type: {label}")
            
if __name__ == "__main__":
    tp = TimePeriod.M60
    print(tp.to_timedelta())