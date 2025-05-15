from enum import Enum
from pathlib import PosixPath
from typing import Any

import numpy as np
from loguru import logger

from common.type import Side
from experts.core.expert import DecisionMaker
from indicators.vol_distribution import VolDistribution


class BuyAndHold(DecisionMaker):
    type = "buy_and_hold"

    def __init__(self, cfg):
        super().__init__(hist_size=cfg.pop("hist_size"),
                         period=cfg.pop("period"),
                         symbol=cfg.pop("symbol"))
        self.indicators = []
        self.description = DecisionMaker.make_description(self.type, cfg)

    def setup_indicators(self, cfg: dict[str, Any]):
        pass

    def look_around(self, h) -> DecisionMaker.Response:


        return DecisionMaker.Response(side=Side.BUY, target_volume_fraction=1)

    def update_inner_state(self, h):
        return None
        