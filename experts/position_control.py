from abc import ABC, abstractmethod

import numpy as np

from common.type import Side
from common.utils import name_from_cfg
from indicators import *

from .base import ExtensionBase


class StopsController(ABC):
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name

    def __str__(self):
        return name_from_cfg(self.cfg, self.name)

    @abstractmethod        
    def _eval_stops(self, common, h) -> tuple:
        pass
    
    def __call__(self, common, h) -> tuple:
        tp, sl = self._eval_stops(common, h)
        if common.active_position.side == Side.BUY:
            if sl:
                sl = np.min([sl, h["Low"][-2], h["Open"][-1]])
        if common.active_position.side == Side.SELL:
            if sl:
                sl = np.max([sl, h["High"][-2], h["Open"][-1]])
        return tp, sl

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
    

class StopsDynamic(StopsController):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsDynamic, self).__init__(cfg, name="stops_dyn")
    
    def _eval_stops(self, common, h):
        tp, sl = None, None
        dir = common.active_position.side.value
        if self.cfg.tp_active:
            tp = -common.order_dir*common.tp[common.order_dir]
        if self.cfg.sl_active:
            sl = common.sl[dir]
        return tp, sl