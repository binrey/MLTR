from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

# import torch
from backtesting.backtest_broker import Position
from common.type import Line, Side

# from experts.position_control import StopsController


def init_target_from_cfg(cfg):
    cfg = deepcopy(cfg)
    Target = cfg.pop("type")
    return Target(cfg)

class ExpertBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.decision_maker: DecisionMaker = init_target_from_cfg(cfg["decision_maker"])
        assert "sl_processor" in cfg, "sl_processor must be defined in cfg"
        self.sl_processor = init_target_from_cfg(cfg["sl_processor"])
        self.sl_processor.set_expert(self)
        self.sl = None
        assert "tp_processor" in cfg, "tp_processor must be defined in cfg"
        self.tp_processor = init_target_from_cfg(cfg.get("tp_processor", None))
        self.tp_processor.set_expert(self)
        self.tp = None
        self.orders = []
            
    def __str__(self):
        return f"{str(self.decision_maker)} sl: {str(self.sl_processor)}  tp: {str(self.tp_processor)}"
    
    @abstractmethod
    def get_body(self) -> None:
        pass
    
    @abstractmethod
    def create_orders(self) -> None:
        pass
    
    def update(self, h, active_position: Position):
        self.active_position = active_position
        self.get_body(h)
        


class DecisionMaker(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
        self.tp_definer = {Side.BUY: None, Side.SELL: None}
        self.lprice, self.sprice, self.cprice, self.tsignal = None, None, None, None
        self.indicator_vis_objects = None
        self.vis_items = []
        self.indicator = self.setup_indicator(cfg)
             
    def __str__(self):
        return self.type + ": " + "|".join([f"{k}:{v}" for k, v in self.cfg.items()])
    
    @abstractmethod
    def setup_indicator(self, cfg):
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
    def look_around(self, h) -> bool:
        pass
    
    @abstractmethod
    def update_inner_state(self, h):
        pass




