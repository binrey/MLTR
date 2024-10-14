import numpy as np

from common.type import Side
from experts.base import DecisionMaker
from indicators.vol_distribution import VolDistribution


class HVOL(DecisionMaker):
    type = "hvol"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.vol_distribution = VolDistribution(cfg)

    def __call__(self, h) -> bool:
        flag = False
        hist_values = self.vol_distribution.update(h)
        max_vol_id = self.vol_distribution.vol_hist.argmax()
        if 0 < max_vol_id < len(hist_values)-1:
            flag = True
        
        lprice, sprice = None, None
        if flag:
            # self.sl_definer[Side.BUY] = h["Low"][:-1].min()
            # self.sl_definer[Side.SELL] = h["High"][:-1].max()         
            lprice = self.vol_distribution.price_bins[max_vol_id+1]
            sprice = self.vol_distribution.price_bins[max_vol_id-1]

        return lprice, sprice
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return None
