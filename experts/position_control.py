from indicators import *

from .base import ExtensionBase


class StopsFixed(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsFixed, self).__init__(cfg, name="stops_fix")
        
    def __call__(self, common, h, sl_custom=None):
        tp = -common.order_dir*h.Open[-1]*(1+common.order_dir*self.cfg.tp*self.cfg.sl/100) if self.cfg.tp is not None else self.cfg.tp
        sl = self.cfg.sl
        if sl_custom is not None:
            sl = sl_custom
        if sl is not None:
            sl = -common.order_dir*h.Open[-1]*(1-common.order_dir*sl/100)
            
        return tp, sl
    

class StopsDynamic(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsDynamic, self).__init__(cfg, name="stops_dyn")
        
    def __call__(self, common, h):
        tp, sl = None, None
        if self.cfg.tp_active:
            tp = -common.order_dir*common.tp[common.order_dir]
        if self.cfg.sl_active:
            sl_add = 0#(common.sl[-1] - common.sl[1])
            sl = -common.order_dir*(common.sl[common.order_dir] - common.order_dir*sl_add)
        return tp, sl