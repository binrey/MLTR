from abc import ABC, abstractmethod

import numpy as np

from common.type import Side
from common.utils import name_from_cfg
from indicators import *

from .base import ExpertBase, ExtensionBase


class StopsController(ABC):
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name

    def __str__(self):
        return name_from_cfg(self.cfg, self.name)

    def set_expert(self, expert):
        self.expert = expert

    @abstractmethod        
    def _eval(self, **kwargs) -> float:
        pass
    
    def __call__(self, h) -> float:
        val = self._eval(hist=h)
        if self.expert.active_position.side == Side.BUY:
            if val:
                val = np.min([val, h["Low"][-2], h["Open"][-1]])
        if self.expert.active_position.side == Side.SELL:
            if val:
                val = np.max([val, h["High"][-2], h["Open"][-1]])
        return val

class StopsFixed(StopsController, ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsFixed, self).__init__(cfg, name="stops_fix")

    def _eval_stops(self, common, h, sl_custom=None):
        dir = common.active_position.side.value
        # tp = -common.order_dir*h.Open[-1]*(1+common.order_dir*self.cfg.tp*self.cfg.sl/100) if self.cfg.tp is not None else self.cfg.tp
        sl, tp = self.cfg.sl, None
        if sl_custom is not None:
            sl = sl_custom
        if sl is not None:
            sl = h["Open"][-1]*(1 - dir*sl/100)
            
        return tp, sl
    

class SLDynamic(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(SLDynamic, self).__init__(cfg, name="sl_dyn")
    
    def _eval(self, **kwargs):
        if self.cfg.sl_active:
            sl = self.expert.body_cls.setup_sl(self.expert.active_position.side)
        return sl