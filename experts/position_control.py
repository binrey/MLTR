from abc import ABC, abstractmethod

import numpy as np

from common.type import Side
from common.utils import name_from_cfg
from indicators import *

from .base import ExpertBase


class StopsController(ABC):
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name

    def __str__(self):
        return name_from_cfg(self.cfg, self.name)

    def set_expert(self, expert: ExpertBase):
        self.expert = expert

    @abstractmethod        
    def _eval(self, **kwargs) -> float:
        pass
    
    def __call__(self, h) -> float:
        val = self._eval(hist=h)
        # if self.expert.active_position.side == Side.BUY:
        #     if val:
        #         val = np.min([val, h["Low"][-2], h["Open"][-1]])
        # if self.expert.active_position.side == Side.SELL:
        #     if val:
        #         val = np.max([val, h["High"][-2], h["Open"][-1]])
        return val
    

class SLDynamic(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(SLDynamic, self).__init__(cfg, name="sl_dyn")
    
    def _eval(self, **kwargs):
        sl = None
        if self.cfg["active"]:
            sl = self.expert.decision_maker.setup_sl(self.expert.active_position.side)
        return sl


class SLFixed(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(SLFixed, self).__init__(cfg, name="sl_fix")
    
    def _eval(self, **kwargs):
        sl = None
        open_price = self.expert.active_position.open_price
        if self.cfg["active"]:
            sl = open_price * (1 - self.cfg["percent_value"] / 100 * self.expert.active_position.side.value)
        return sl
    

class TPFromSL(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(TPFromSL, self).__init__(cfg, name="tp_from_sl")
    
    def _eval(self, **kwargs):
        tp = None
        open_price = self.expert.active_position.open_price
        if self.cfg["active"]:
            tp = open_price + self.cfg["scale"] * abs(open_price - self.expert.active_position.sl) * self.expert.active_position.side.value
        return tp
    
    
def fix_rate_trailing_sl(sl:float,
                       open_price: float,
                       side: Side,
                       trailing_stop_rate:float,
                       ticksize: float) -> float:
    sl_new = float(sl + trailing_stop_rate*(open_price - sl))
    return sl_new