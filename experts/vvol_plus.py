from enum import Enum

import numpy as np
from loguru import logger

from common.type import Side
from experts.core.expert import DecisionMaker
from indicators.vol_distribution import VolDistribution
from indicators.zigzag import ZigZag


class VVolPlus(DecisionMaker):
    type = "vvol_plus"

    class TriggerStrategy(str, Enum):
        MANUAL_LEVELS = "manual"  # Uses predefined volume distribution bins
        AUTO_LEVELS = "auto"  # Uses shadow intersection analysis

    def __init__(self, cfg):
        super().__init__(cfg)
        self.long_bin = cfg.get("long_bin", 1)
        self.short_bin = cfg.get("short_bin", 1)
        self.sharpness = cfg["sharpness"]
        self.strategy = cfg["strategy"]

    def setup_indicators(self, cfg):
        self.indicator = VolDistribution(nbins=cfg["nbins"])
        self.zigzag = ZigZag(cfg["zigzag_period"])
        return [self.indicator, self.zigzag]

    def _find_prices_manual_levels(self, h, max_vol_id):
        """Manual levels strategy: Uses volume distribution bins directly"""
        if self.short_bin <= max_vol_id < len(self.indicator.vol_hist) - self.long_bin:
            bin_width = self.indicator.price_bins[1] - \
                self.indicator.price_bins[0]
            lprice = self.indicator.price_bins[max_vol_id +
                                               self.long_bin] + bin_width/2
            sprice = self.indicator.price_bins[max_vol_id -
                                               self.short_bin] + bin_width/2
            # if sprice < h["Open"][-1] < lprice:
            return lprice, sprice
        return None, None

    def _find_prices_auto_levels(self, h):
        """Auto levels strategy: Uses shadow intersection analysis"""
        potential_prices = np.linspace(h["Low"].min(), h["High"].max(), 100)
        upper_scores = np.zeros(len(potential_prices))
        bottom_scores = np.zeros(len(potential_prices))

        for i, price in enumerate(potential_prices):
            upper_shadow_hits = np.sum((h["High"] >= price) & (
                np.maximum(h["Open"], h["Close"]) <= price))
            bottom_shadow_hits = np.sum(
                (np.minimum(h["Open"], h["Close"]) >= price) & (h["Low"] <= price))

            body_hits = np.sum(
                (np.minimum(h["Open"], h["Close"]) <= price) &
                (np.maximum(h["Open"], h["Close"]) >= price)
            )

            upper_scores[i] = upper_shadow_hits - body_hits
            bottom_scores[i] = bottom_shadow_hits - body_hits

        lprice = potential_prices[np.argmax(upper_scores)]
        sprice = potential_prices[np.argmax(bottom_scores)]

        if sprice < h["Open"][-1] < lprice:
            return lprice, sprice
        return None, None

    def look_around(self, h) -> DecisionMaker.Response:
        order_side, target_volume_fraction = None, 1
        self.indicator.update(h)
        self.zigzag.update(h)
        max_vol_id = self.indicator.vol_hist.argmax()

        if self.indicator.vol_hist[max_vol_id] / self.indicator.vol_hist.mean() > self.sharpness:
            if self.strategy == self.TriggerStrategy.MANUAL_LEVELS:
                self.lprice, self.sprice = self._find_prices_manual_levels(h, max_vol_id)
            elif self.strategy == self.TriggerStrategy.AUTO_LEVELS:
                self.lprice, self.sprice = self._find_prices_auto_levels(h)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            if self.lprice is not None and self.sprice is not None:
                self.sl_definer[Side.BUY] = h["Low"].min()
                self.sl_definer[Side.SELL] = h["High"].max()
                self.set_draw_objects(h["Date"][-2])
                self.draw_items += self.indicator.vis_objects

        if self.lprice:
            zz_values, zz_types = self.zigzag.values, self.zigzag.types
            if zz_values[-2] > self.lprice and zz_types[-2] < 0 and h["Close"][-2] > self.lprice:
                order_side = Side.BUY
                self.draw_items += self.zigzag.vis_objects
                self._reset_state()

        if self.sprice:
            zz_values, zz_types = self.zigzag.values, self.zigzag.types
            if zz_values[-2] < self.sprice and zz_types[-2] > 0 and h["Close"][-2] < self.sprice:
                order_side = Side.SELL
                self.draw_items += self.zigzag.vis_objects
                self._reset_state()
                
        if self.lprice or self.sprice:
            logger.debug(
                f"found enter points: long: {self.lprice}, short: {self.sprice}")

        return DecisionMaker.Response(side=order_side,
                                      target_volume_fraction=target_volume_fraction)

    def setup_sl(self, side: Side) -> float:
        return self.sl_definer[side]

    def setup_tp(self, side: Side) -> float:
        return self.tp_definer[side]

    def update_inner_state(self, h) -> None:
        return super().update_inner_state(h)
