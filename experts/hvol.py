import numpy as np

from common.type import Side
from experts.base import DecisionMaker
from indicators.vol_distribution import VolDistribution


class HVOL(DecisionMaker):
    type = "hvol"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trigger_width = 1
        
    def setup_indicator(self, cfg):
        return VolDistribution(nbins=cfg["nbins"])

    def look_around(self, h) -> bool:
        flag = False
        self.indicator.update(h)
        max_vol_id = self.indicator.vol_hist.argmax()
        if self.trigger_width <= max_vol_id < len(self.indicator.vol_hist) - self.trigger_width:
            if self.indicator.vol_hist[max_vol_id] / self.indicator.vol_hist.mean() > self.cfg["sharpness"]:
                bin_width = self.indicator.price_bins[1] - self.indicator.price_bins[0] 
                lprice = self.indicator.price_bins[max_vol_id+self.trigger_width] + bin_width/2
                sprice = self.indicator.price_bins[max_vol_id-self.trigger_width] + bin_width/2
                if sprice < h["Open"][-1] < lprice:
                    flag = True
        
        # lprice, sprice = None, None
        if flag:
            self.lprice = lprice
            self.sprice = sprice
            self.tsignal = None#h["Date"][-2]
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()
            self.indicator_vis_objects = self.indicator.vis_objects
        return flag
    
    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)