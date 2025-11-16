from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Optional

from pandas import Period

from common.type import Line, Side, SLDefiner, Symbol, TPDefiner, to_datetime
from common.utils import set_indicator_cache_dir


class DecisionMaker(ABC):
    @dataclass
    class Response:
        side: Side | None
        target_volume_fraction: Optional[float] = None
        increment_volume_fraction: Optional[float] = None
        increment_by_num_lots: Optional[int] = None

        # ensure at most one of: target_volume_fraction, increment_volume_fraction, increment_by_num_lots
        def __post_init__(self):
            num_specified = sum(
                x is not None
                for x in (
                    self.target_volume_fraction,
                    self.increment_volume_fraction,
                    self.increment_by_num_lots,
                )
            )
            if num_specified > 1:
                raise ValueError(
                    "Only one of target_volume_fraction, increment_volume_fraction, increment_by_num_lots can be not None"
                )

        @property
        def is_active(self):
            return self.side is not None

    def __init__(self, cfg: dict[str, Any]):
        self.hist_size: int = cfg.pop("hist_size")
        self.period: Period = cfg.pop("period")
        self.symbol: Symbol = cfg.pop("symbol")

        self.sl_definer: SLDefiner = {Side.BUY: None, Side.SELL: None}
        self.tp_definer: TPDefiner = {Side.BUY: None, Side.SELL: None}
        self.lprice, self.sprice, self.cprice = None, None, None
        self.draw_items = []
        self.cache_dir = set_indicator_cache_dir(self.symbol, self.period, self.hist_size)
        self.description = None

    @staticmethod
    def make_description(ds_type, cfg):
        return ds_type + ": " + "|".join([f"{k}:{v}" for k, v in cfg.items()])

    def __str__(self):
        return self.description

    @abstractmethod
    def setup_indicators(self, cfg, indicator_cache_dir: PosixPath):
        pass

    def set_draw_objects(self, time=None):
        self.draw_items = []
        if self.lprice is not None and time:
            self.draw_items.append(
                Line([(to_datetime(time), self.lprice), (None, self.lprice)], color="green"))
        if self.sprice is not None and time:
            self.draw_items.append(
                Line([(to_datetime(time), self.sprice), (None, self.sprice)], color="red"))

        # for indicator in self.indicators:
        #     self.draw_items += indicator.vis_objects

    @abstractmethod
    def look_around(self, h) -> "DecisionMaker.Response":
        pass

    @abstractmethod
    def update_inner_state(self, h) -> None:
        pass

    def _reset_state(self):
        """Reset all state variables to their initial values"""
        self.lprice = None
        self.sprice = None
        self.cprice = None
        # self.sl_definer.update({Side.BUY: None, Side.SELL: None})
        # self.tp_definer.update({Side.BUY: None, Side.SELL: None})
