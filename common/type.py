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
        if side == 1:
            return Side.BUY
        elif side == -1:
            return Side.SELL
        else:
            raise ValueError(f"{side} is not valid value, set [1 or -1]")

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

    def to_timedelta(self):
        return np.timedelta64(self.value[1:], self.value[0].lower())
    
    @property
    def minutes(self):
        return int(self.value[1:])
        

if __name__ == "__main__":
    tp = TimePeriod.M60
    print(tp.to_timedelta())