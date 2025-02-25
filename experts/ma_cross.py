from pathlib import Path

import torch
from indicators.ma import MovingAverage
from common.type import Side
from .core.expert import DecisionMaker


class ClsMACross(DecisionMaker):
    type = "macross"

    def __init__(self, cfg):
        super(ClsMACross, self).__init__(cfg)
        self.mode = self.cfg["mode"]
        self.model = None
        if Path("model.pt").exists():
            self.model = torch.jit.load("model.pt")

    def setup_indicators(self, cfg):
        self.ma_fast = MovingAverage(period=cfg["ma_fast_period"])
        self.ma_slow = MovingAverage(
            period=cfg["ma_slow_period"],
            upper_levels=cfg["upper_levels"],
            lower_levels=cfg["lower_levels"],
            min_step=cfg["min_step"],
            speed=cfg["speed"])
        return self.ma_fast

    def look_around(self, h) -> bool:
        order_side, lots_to_order, volume_fraction = None, None, None
        self.ma_fast.update(h)
        self.ma_slow.update(h)
        levels_curr, levels_prev = self.ma_slow.current_ma_values, self.ma_slow.previous_ma_values
        ma_slow_prev, ma_slow_curr = levels_prev[0], levels_curr[0]
        ma_fast_prev, ma_fast_curr = self.ma_fast.previous_ma_values[0], self.ma_fast.current_ma_values[0]
        
        if self.mode == "trend":
            if ma_fast_curr > ma_slow_curr:
                order_side = Side.BUY
                volume_fraction = 1
            elif ma_fast_curr < ma_slow_curr:
                order_side = Side.SELL
                volume_fraction = 1

        elif self.mode == "contrtrend":
            if ma_slow_curr and ma_slow_prev:
                for level in range(self.ma_slow.upper_levels, -1, -1): #self.ma_slow.levels_count//2, 0, -1):
                    if levels_curr[level]:
                        if ma_fast_prev > levels_prev[level] and ma_fast_curr < levels_curr[level]:
                            order_side = Side.SELL
                            lots_to_order = abs(level)
                            break
                for level in range(-self.ma_slow.lower_levels, 1, 1):
                    if levels_curr[level]:
                        if ma_fast_prev < levels_prev[level] and ma_fast_curr > levels_curr[level]:
                            order_side = Side.BUY
                            lots_to_order = abs(level)
                            break
        else:
            raise Exception("Unknown mode")         

        if order_side:
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()
            self.indicator_vis_objects = self.ma_fast.get_vis_objects() + self.ma_slow.get_vis_objects()

        return DecisionMaker.Response(side=order_side, 
                                      target_volume_fraction=volume_fraction,
                                      increment_by_num_lots = lots_to_order)
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)
