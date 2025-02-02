from pathlib import Path

import torch
from indicators.ma import MovingAverage
from common.type import Side
from .core.expert import *


class ClsMACross(DecisionMaker):
    type = "macross"

    def __init__(self, cfg):
        super(ClsMACross, self).__init__(cfg)
        self.mode = self.cfg["mode"]
        self.model = None
        if Path("model.pt").exists():
            self.model = torch.jit.load("model.pt")

    def setup_indicators(self, cfg):
        self.ma_fast = MovingAverage(cfg["ma_fast_period"])
        self.ma_slow = MovingAverage(
            cfg["ma_slow_period"], 
            levels_count=cfg["levels_count"],
            levels_step=cfg["levels_step"])
        return self.ma_fast

    def look_around(self, h) -> bool:
        target_volume_fraction = 0
        ma_fast_value = self.ma_fast.update(h)
        ma_slow_value = self.ma_slow.update(h)
        
        if self.mode == "trend":
            if ma_fast_value > ma_slow_value:
                self.lprice = h["Open"][-1]
                self.sprice = None
                target_volume_fraction = 1
            elif ma_fast_value < ma_slow_value:
                self.lprice = None 
                self.sprice = h["Open"][-1]
                target_volume_fraction = 1

        elif self.mode == "contrtrend":
            levels_values = self.ma_slow.last_ma_values
            ma_fast_last, ma_fast_curr = self.ma_fast.main_ma_values[-2:]
            if ma_fast_last > 0 and ma_fast_curr > 0:
                for level in range(self.ma_slow.levels_count//2, 0, -1):
                    if ma_fast_last > levels_values[level] and ma_fast_curr < levels_values[level]:
                        self.lprice = None 
                        self.sprice = h["Open"][-1]
                        target_volume_fraction = level/self.ma_slow.levels_count*2
                        break
                for level in range(-self.ma_slow.levels_count//2, 0, 1):
                    if ma_fast_last < levels_values[level] and ma_fast_curr > levels_values[level]:
                        self.lprice = h["Open"][-1]
                        self.sprice = None
                        target_volume_fraction = abs(level)/self.ma_slow.levels_count*2
                        break
        else:
            raise Exception("Unknown mode")         

        if target_volume_fraction > 0:
            self.tsignal = None
            self.target_volume_fraction = target_volume_fraction
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()
            self.indicator_vis_objects = self.ma_fast.get_vis_objects() + self.ma_slow.get_vis_objects()
        return self.lprice, self.sprice, self.cprice
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)
