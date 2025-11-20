
from typing import Any
from common.type import Side
from indicators.ma import MovingAverage

from .core.expert import DecisionMaker


class ClsMACross(DecisionMaker):
    type = "macross"

    def __init__(self, cfg: dict[str, Any]):
        super(ClsMACross, self).__init__(cfg)
        self.ma_slow_period = self.hist_size
        self.ma_fast_period = self.ma_slow_period//cfg["ma_fast_period"]
        self.upper_levels = cfg["upper_levels"]
        self.lower_levels = cfg["lower_levels"]
        self.min_step = cfg["min_step"]
        self.speed = self.min_step #cfg["speed"]
        self.indicators = self.setup_indicators()
        self.description = DecisionMaker.make_description(self.type, cfg)


    def setup_indicators(self):
        self.ma_fast = MovingAverage(period=self.ma_fast_period)
        self.ma_slow = MovingAverage(
            period=self.ma_slow_period,
            upper_levels=self.upper_levels,
            lower_levels=self.lower_levels,
            min_step=self.min_step,
            speed=self.speed)
        return [self.ma_fast, self.ma_slow]

    def look_around(self, h) -> bool:
        order_side, lots_to_order = None, None
        self.ma_fast.update(h)
        self.ma_slow.update(h)
        levels_curr, levels_prev = self.ma_slow.current_ma_values, self.ma_slow.previous_ma_values
        ma_slow_prev, ma_slow_curr = levels_prev[0], levels_curr[0]
        ma_fast_prev, ma_fast_curr = self.ma_fast.previous_ma_values[0], self.ma_fast.current_ma_values[0]

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

        if order_side:
            self.sl_definer[Side.BUY] = h["Low"][-self.ma_fast_period:].min()
            self.sl_definer[Side.SELL] = h["High"][-self.ma_fast_period:].max()
            self.set_draw_objects(h["Date"][-2])
            self.draw_items += self.ma_slow.vis_objects
            self.draw_items += self.ma_fast.vis_objects

        response = DecisionMaker.Response(side=order_side,
                                          increment_by_num_lots=lots_to_order)
        return response

    def setup_sl(self, side: Side):
        return self.sl_definer[side]

    def setup_tp(self, side: Side):
        return self.tp_definer[side]

    def update_inner_state(self, h):
        return super().update_inner_state(h)
