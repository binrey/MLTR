from pathlib import Path

import torch
from indicators.ma import MovingAverage
from common.type import Side
from .core.expert import *


class ClsMACross(DecisionMaker):
    type = "macross"

    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsMACross, self).__init__(cfg)
        self.model = None
        if Path("model.pt").exists():
            self.model = torch.jit.load("model.pt")

    def setup_indicator(self, cfg):
        self.ma_fast = MovingAverage(cfg["ma_fast_period"])
        self.ma_slow = MovingAverage(cfg["ma_slow_period"])
        return self.ma_fast

    def look_around(self, h) -> bool:
        flag = False
        self.ma_fast.update(h)
        self.ma_slow.update(h)
        
        ma_fast_value = self.ma_fast.values[-1]
        ma_slow_value = self.ma_slow.values[-1]
        
        if ma_fast_value > ma_slow_value:
            self.lprice = h["Open"][-1]
            self.sprice = None
            flag = True
        elif ma_fast_value < ma_slow_value:
            self.lprice = None 
            self.sprice = h["Open"][-1]
            flag = True
            
        if flag:
            self.tsignal = None
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()
            self.indicator_vis_objects = self.ma_fast.get_vis_objects() + self.ma_slow.get_vis_objects()
        return flag
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)
