from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from common.type import Side
from experts.core.decision_maker import DecisionMaker


class Random(DecisionMaker):
    type = "random"
    
    class Levels(str, Enum):
        """Strategy type for HVOL price level detection"""
        MANUAL = "manual"  # Uses predefined volume distribution bins
        AUTO = "auto"  # Uses shadow intersection analysis

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.seed = cfg["seed"]
        self.time_to_wait = cfg["time_to_wait"]
        np.random.seed(self.seed)
        self.timer = self.time_to_wait

    def look_around(self, h) -> DecisionMaker.Response:
        order_side, target_volume_fraction = None, 1

        # if self.timer < self.time_to_wait:
        #     self.timer += 1
        #     order_side = None
        # else:
        #     order_side = np.random.choice([Side.BUY, Side.SELL, None, None])
        #     self.timer = 0

        if order_side is not None:
            self.sl_definer[Side.BUY] = h["Low"].min()
            self.sl_definer[Side.SELL] = h["High"].max()

        logger.debug(f"order_side: {order_side}")
        return DecisionMaker.Response(side=order_side, target_volume_fraction=target_volume_fraction)

    def setup_sl(self, side: Side):
        return self.sl_definer[side]

    def setup_tp(self, side: Side):
        return self.tp_definer[side]
    
    def setup_indicators(self, cfg: dict[str, Any]):
        return {}

    def update_inner_state(self, h):
        return super().update_inner_state(h)
        