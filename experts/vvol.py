import numpy as np
from common.type import Side
from experts.base import DecisionMaker


class VVOL(DecisionMaker):
    type = "vvol"
    
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, h) -> bool:
        is_fig = False
        best_params = {
            "metric": 0,
            "i": 0,
            "line_above": None,
            "line_below": None,
        }

        x_bins = np.linspace(h["Low"][:-1].min(), h["High"][:-1].max(), 10)
        x = (h["High"][:-1] + h["Low"][:-1])/2
        y = h["Volume"][:-1]
        hist = np.histogram(x, bins=x_bins, weights=y)[0]
        line = x_bins[hist.argmax()]
        is_fig = True
        
        lprice, sprice = None, None
        if is_fig:
            i = best_params["i"]
            self.sl_definer[Side.BUY] = h["Low"][:-1].min()
            self.sl_definer[Side.SELL] = h["High"][:-1].max()         
            lprice = x_bins[hist.argmax()+1]
            sprice = x_bins[hist.argmax()-1]

        return lprice, sprice
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return None
