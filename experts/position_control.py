from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

from common.utils import name_from_cfg
from experts.core.expert import DecisionMaker
from trade.utils import Position


class StopsController(ABC):
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name

    def __str__(self):
        return name_from_cfg(self.cfg, self.name)
    
    @abstractmethod
    def create(self, 
               active_position: Position,
               hist: Optional[np.ndarray] = None,
               decision_maker: Optional[DecisionMaker] = None) -> float:
        pass

    
class SLDynamic(StopsController):
    def __init__(self, cfg):
        self.active = cfg["active"]
        super(SLDynamic, self).__init__(cfg, name="sl_dyn")
    
    def create(self, **kwargs):
        decision_maker = kwargs["decision_maker"]
        active_position = kwargs["active_position"]
        sl = None
        if self.active and active_position is not None:
            sl = decision_maker.setup_sl(active_position.side)
        return sl


class SLFixed(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(SLFixed, self).__init__(cfg, name="sl_fix")
    
    def create(self, **kwargs):
        active_position = kwargs["active_position"]
        sl = None
        open_price = active_position.open_price
        if self.cfg["active"]:
            sl = open_price * (1 - self.cfg["percent_value"] / 100 * active_position.side.value)
        return sl
    

class TPFromSL(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(TPFromSL, self).__init__(cfg, name="tp_from_sl")
    
    def create(self, **kwargs):
        active_position = kwargs["active_position"]
        tp = None
        open_price = active_position.open_price
        if self.cfg["active"] and active_position.sl is not None:
            tp = open_price + self.cfg["scale"] * abs(open_price - active_position.sl) * active_position.side.value
        return tp


class TrailingStopStrategy(Enum):
    FIX_RATE = "fix_rate"

class TrailingStop:
    FIX_RATE = "fix_rate"
    MA_ACCELERATED = "ma_accelerated"
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def get_stop_loss(self, active_position: Position, hist: np.ndarray):
        return {self.FIX_RATE: self.fix_rate_trailing_sl(active_position, hist, self.cfg["trailing_stop_rate"]),
                self.MA_ACCELERATED: self.ma_accelerated_trailing_sl(active_position, hist)
                }[self.cfg["strategy"]]
        
    def fix_rate_trailing_sl(self, active_position: Position, hist: np.ndarray, rate: float) -> float:
        last_price = hist["Open"][-1]
        sl_new = float(active_position.sl + rate*(last_price - active_position.sl))
        return sl_new
    
    def ma_accelerated_trailing_sl(self, active_position: Position, hist: np.ndarray) -> float:
        ma_curr = hist["Close"][-8:].mean()
        ma_prev = hist["Close"][-9:-1].mean()
        dma = abs(ma_curr - ma_prev) / ma_prev * 100
        
        acceleration = max((dma // 1) * 1, 0) * 0.1
        sl_new = self.fix_rate_trailing_sl(active_position, hist, self.cfg["trailing_stop_rate"] * acceleration)
        return sl_new