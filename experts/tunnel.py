from common.type import Side
from experts.base import DecisionMaker


class ClsTunnel(DecisionMaker):
    type = "tunnel"
    
    def __init__(self, cfg):
        super(ClsTunnel, self).__init__(cfg)

    def look_around(self, h) -> bool:
        is_fig = False
        best_params = {
            "metric": 0,
            "i": 0,
            "line_above": None,
            "line_below": None,
        }
        for i in range(4, h.shape[0], 1):
            line_above = h["High"][-i:-1].mean()
            line_below = h["Low"][-i:-1].mean()
            middle_line = (line_above + line_below) / 2

            if h["Close"][-2] < line_above and h["Close"][-2] > line_below:
                metric = i / ((line_above - line_below) / middle_line) / 100

                if metric > best_params["metric"]:
                    best_params.update(
                        {"metric": metric,
                        "i": i,
                        "line_above": line_above,
                        "line_below": line_below,
                        "middle_line": middle_line
                        }
                    )

        if best_params["metric"] > self.cfg["ncross"]:
            is_fig = True
            # break

        lprice, sprice = None, None
        if is_fig:
            i = best_params["i"]
            self.sl_definer[Side.BUY] = h["Low"][-i:-1].min()
            self.sl_definer[Side.SELL] = h["High"][-i:-1].max()         
            self.lprice = best_params["line_above"]
            self.sprice = best_params["line_below"]

        return is_fig
    
    def setup_indicator(self, cfg):
        pass
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return None

    def update_inner_state(self, h):
        return super().update_inner_state(h)