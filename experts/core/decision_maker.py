from abc import ABC, abstractmethod
import pandas as pd
from common.type import Line, Side
from dataclasses import dataclass


class DecisionMaker(ABC):
    @dataclass
    class Response:
        side: Side | None
        volume_fraction: float = 1.0

        @property
        def is_active(self):
            return self.side is not None

    def __init__(self, cfg):
        self.cfg = cfg
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
        self.tp_definer = {Side.BUY: None, Side.SELL: None}
        self.lprice, self.sprice, self.cprice = None, None, None
        self.indicator_vis_objects = None
        self.vis_items = []
        self.setup_indicators(cfg)
             
    def __str__(self):
        return self.type + ": " + "|".join([f"{k}:{v}" for k, v in self.cfg.items()])
    
    @abstractmethod
    def setup_indicators(self, cfg):
        pass
    
    @property
    def vis_objects(self):
        lines = []
        if self.lprice is not None:
            lines.append(Line([(pd.to_datetime(self.tsignal), self.lprice), (None, self.lprice)], color="green"))
        if self.sprice is not None:
            lines.append(Line([(pd.to_datetime(self.tsignal), self.sprice), (None, self.sprice)], color="red"))
        if self.indicator_vis_objects is not None:
            lines += self.indicator_vis_objects
        return lines
      
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


