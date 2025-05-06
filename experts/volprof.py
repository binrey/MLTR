from enum import Enum
from pathlib import PosixPath
from typing import Any

import numpy as np
from loguru import logger

from common.type import Side
from experts.core.expert import DecisionMaker
from indicators.vol_distribution import VolDistribution


class VolProf(DecisionMaker):
    type = "volprof"
    
    class Levels(str, Enum):
        """Strategy type for HVOL price level detection"""
        MANUAL = "manual"  # Uses predefined volume distribution bins
        AUTO = "auto"  # Uses shadow intersection analysis
    
    def __init__(self, cfg):
        super().__init__(cfg["hist_size"], cfg["period"], cfg["symbol"])
        cfg.pop("hist_size")
        cfg.pop("period")
        cfg.pop("symbol")
        self.indicators = self.setup_indicators(cfg)
        
        self.long_bin = 1
        self.short_bin = 1
        self.sharpness = cfg["sharpness"]
        self.strategy = cfg["strategy"]
        
        self.description = DecisionMaker.make_description(self.type, cfg)

    def setup_indicators(self, cfg: dict[str, Any]):
        self.indicator = VolDistribution(cache_dir=self.cache_dir)
        return [self.indicator]

    def _find_prices_manual_levels(self, h, max_vol_id):
        """Manual levels strategy: Uses volume distribution bins directly"""
        if self.short_bin <= max_vol_id < len(self.indicator.vol_hist) - self.long_bin:
            lprice = self.indicator.price_bins[max_vol_id + self.long_bin]
            lprice += self.indicator.bin_size/2
            sprice = self.indicator.price_bins[max_vol_id - self.short_bin]
            sprice += self.indicator.bin_size/2
            return lprice, sprice
        return None, None

    def _find_prices_auto_levels(self, h):
        """Auto levels strategy: Uses shadow intersection analysis"""
        potential_prices = np.linspace(h["Low"].min(), h["High"].max(), 100)
        upper_scores = np.zeros(len(potential_prices))
        bottom_scores = np.zeros(len(potential_prices))
        
        for i, price in enumerate(potential_prices):
            upper_shadow_hits = np.sum((h["High"] >= price) & (np.maximum(h["Open"], h["Close"]) <= price))
            bottom_shadow_hits = np.sum((np.minimum(h["Open"], h["Close"]) >= price) & (h["Low"] <= price))
            
            body_hits = np.sum(
                (np.minimum(h["Open"], h["Close"]) <= price) &
                (np.maximum(h["Open"], h["Close"]) >= price)
            )
            
            upper_scores[i] = upper_shadow_hits - body_hits
            bottom_scores[i] = bottom_shadow_hits - body_hits
        
        lprice = potential_prices[np.argmax(upper_scores)]
        sprice = potential_prices[np.argmax(bottom_scores)]
        
        logger.debug(f"check condition: sprice ({sprice:.2f}) < Open ({h['Open'][-1]:.2f}) < lprice ({lprice:.2f})")
        if sprice < h["Open"][-1] < lprice:
            return lprice, sprice
        return None, None

    def look_around(self, h) -> DecisionMaker.Response:
        order_side, target_volume_fraction = None, 1
        self.indicator.update(h)
        max_vol_id = self.indicator.vol_hist.argmax()

        if self.indicator.vol_hist[max_vol_id] / self.indicator.vol_hist.mean() > self.sharpness:
            if self.strategy == self.Levels.MANUAL:
                self.lprice, self.sprice = self._find_prices_manual_levels(h, max_vol_id)
            else:  # AUTO_LEVELS
                self.lprice, self.sprice = self._find_prices_auto_levels(h)

            if self.lprice is not None and self.sprice is not None:
                self.sl_definer[Side.BUY] = self.sprice
                self.sl_definer[Side.SELL] = self.lprice
                self.set_draw_objects(h["Date"][-2])
                self.draw_items += self.indicator.vis_objects

        strike = h["Close"][-2] - h["Open"][-2]
        max_body = max(np.maximum(h["Open"][:-2], h["Close"][:-2]) - np.minimum(h["Open"][:-2], h["Close"][:-2]))
        logger.debug(f"check condition curr. body ({abs(strike):.2f}) > max. body ({max_body:.3f})")
        if abs(strike) > max_body:
            if self.lprice is not None:
                if strike > 0 and h["Close"][-2] > self.sprice:
                    order_side = Side.BUY

            if self.sprice is not None:
                if strike < 0 and h["Close"][-2] < self.lprice:
                    order_side = Side.SELL

        if self.lprice or self.sprice:
            logger.debug(f"found enter points: long: {self.lprice}, short: {self.sprice}")

        order_side = np.random.choice([Side.BUY, Side.SELL, None])
        logger.debug(f"order_side: {order_side}")

        return DecisionMaker.Response(side=order_side, target_volume_fraction=target_volume_fraction)

    def setup_sl(self, side: Side):
        return self.sl_definer[side]
    
    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def update_inner_state(self, h):
        return super().update_inner_state(h)
        