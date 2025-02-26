from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from common.type import Line, Side, TimeVolumeProfile


class DecisionMaker(ABC):
    @dataclass
    class Response:
        side: Side | None
        target_volume_fraction: Optional[float] = None #  Define position volume have to be
        increment_volume_fraction: Optional[float] = None #  How mutch increase position, target isn't defined
        increment_by_num_lots: Optional[int] = None

        @property
        def is_active(self):
            return self.side is not None

    def __init__(self, cfg):
        self.cfg = cfg
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
        self.tp_definer = {Side.BUY: None, Side.SELL: None}
        self.lprice, self.sprice, self.cprice = None, None, None
        self.vis_items = []
        self.indicators = self.setup_indicators(cfg)
             
    def __str__(self):
        return self.type + ": " + "|".join([f"{k}:{v}" for k, v in self.cfg.items()])
    
    @abstractmethod
    def setup_indicators(self, cfg):  # TODO
        pass
    
    def set_vis_objects(self, time=None):
        self.draw_items = []
        if self.lprice is not None and time:
            self.draw_items.append(Line([(pd.to_datetime(time), self.lprice), (None, self.lprice)], color="green"))
        if self.sprice is not None and time:
            self.draw_items.append(Line([(pd.to_datetime(time), self.sprice), (None, self.sprice)], color="red"))
            
        for indicator in self.indicators:
            self.draw_items += indicator.vis_objects
    
    @abstractmethod
    def look_around(self, h) -> "DecisionMaker.Response":
        pass
    
    @abstractmethod
    def update_inner_state(self, h):
        pass

    def _reset_state(self):
        """Reset all state variables to their initial values"""
        self.lprice = None
        self.sprice = None
        self.cprice = None
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
        self.tp_definer = {Side.BUY: None, Side.SELL: None}    


