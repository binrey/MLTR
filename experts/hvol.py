import numpy as np

from common.type import Side
from experts.base import DecisionMaker
from indicators.vol_distribution import VolDistribution


class HVOL(DecisionMaker):
    type = "hvol"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def setup_indicator(self, cfg):
        return VolDistribution(nbins=cfg["nbins"])

    def look_around(self, h) -> bool:
        flag = False
        self.indicator.update(h)
        max_vol_id = self.indicator.vol_hist.argmax()
        if 0 < max_vol_id < len(self.indicator.vol_hist) - 1:
            if self.indicator.vol_hist[max_vol_id] / self.indicator.vol_hist.mean() > self.cfg["sharpness"]:
                flag = True
        
        # lprice, sprice = None, None
        if flag:
            bin_width = self.indicator.price_bins[1] - self.indicator.price_bins[0]      
            self.lprice = self.indicator.price_bins[max_vol_id+1]+bin_width/2
            self.sprice = self.indicator.price_bins[max_vol_id-1]+bin_width/2
            self.tsignal = None#h["Date"][-2]
            self.sl_definer[Side.BUY] = self.sprice
            self.sl_definer[Side.SELL] = self.lprice
        return flag
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return None
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)