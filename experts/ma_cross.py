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
        self.ma_fast = MovingAverage(
            cfg["ma_fast_period"],
            levels_count=cfg["levels_count"],
            levels_step=cfg["levels_step"])
        self.ma_slow = MovingAverage(
            cfg["ma_slow_period"])
        return self.ma_fast

    def look_around(self, h) -> bool:
        order_side, target_volume_fraction = None, 0
        self.ma_fast.update(h)
        self.ma_slow.update(h)
        levels_curr, levels_prev = self.ma_fast.current_ma_values, self.ma_fast.previous_ma_values
        ma_fast_prev, ma_fast_curr = levels_prev[0], levels_curr[0]
        ma_slow_curr = self.ma_slow.current_ma_values[0]
        close_prev, close_curr = h["Close"][-3:-1]
        
        if self.mode == "trend":
            if ma_fast_curr > ma_slow_curr:
                oreder_side = Side.BUY
                target_volume_fraction = 1
            elif ma_fast_curr < ma_slow_curr:
                oreder_side = Side.SELL
                target_volume_fraction = 1

        elif self.mode == "contrtrend":

            if ma_fast_curr and ma_fast_prev:
                # if ma_fast_curr < ma_slow_curr:
                for level in range(self.ma_fast.levels_count//2, 0, -1):
                    if levels_curr[level]:
                        if close_prev > levels_prev[level] and close_curr < levels_curr[level]:
                            order_side = Side.SELL
                            target_volume_fraction = level/self.ma_fast.levels_count*2
                            break
                # if ma_fast_curr > ma_slow_curr:
                for level in range(-self.ma_fast.levels_count//2, 0, 1):
                    if levels_curr[level]:
                        if close_prev < levels_prev[level] and close_curr > levels_curr[level]:
                            order_side = Side.BUY
                            target_volume_fraction = abs(level)/self.ma_fast.levels_count*2
                            break
        else:
            raise Exception("Unknown mode")         

        if order_side:
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()
            self.indicator_vis_objects = self.ma_fast.get_vis_objects() + self.ma_slow.get_vis_objects()

        return DecisionMaker.Response(
            side=order_side,
            volume_fraction=target_volume_fraction)
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)
