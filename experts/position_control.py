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
    
    def _eval(self, **kwargs):
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
        if self.cfg["active"]:
            tp = open_price + self.cfg["scale"] * abs(open_price - active_position.sl) * active_position.side.value
        return tp
    
    
def fix_rate_trailing_sl(sl:float,
                       open_price: float,
                       trailing_stop_rate:float) -> float:
    sl_new = float(sl + trailing_stop_rate*(open_price - sl))
    return sl_new


class TrailingStopStrategy(Enum):
    FIX_RATE = "fix_rate"

class TrailingStop:
    FIX_RATE = "fix_rate"
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def get_stop_loss(self, open_price: float):
        return {self.FIX_RATE: self.fix_rate_trailing_sl(open_price)}[self.cfg["trailing_stop"]["strategy"]]
        
    def fix_rate_trailing_sl(self, open_price: float) -> float:
        trailing_stop_rate = self.cfg["trailing_stop_rate"]
        sl_new = float(self.sl + trailing_stop_rate*(open_price - self.sl))
        return sl_new