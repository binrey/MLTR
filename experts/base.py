from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from loguru import logger

# import torch
from backtesting.backtest_broker import Position
from common.type import Side


def init_target_from_cfg(cfg):
    Target = cfg.pop("type")
    return Target(cfg)

class ExpertBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.body_cls = init_target_from_cfg(cfg["body_classifier"])
        self.sl_processor = init_target_from_cfg(cfg["sl_processor"])
        self.sl_processor.set_expert(self)
        
        self.orders = []
            
    def __str__(self):
        return f"{str(self.body_cls)} sl: {str(self.sl_processor)}"
    
    @abstractmethod
    def get_body(self) -> None:
        pass
    
    @abstractmethod
    def create_orders(self) -> None:
        pass
    
    def update(self, h, active_position: Position):
        self.active_position = active_position
        self.get_body(h)
        


class DecisionMaker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
             
    def __str__(self):
        return self.type + ": " + "|".join([f"{k}:{v}" for k, v in self.cfg.items()])

    def __call__(self, common, h):
        pass
            
    def update_inner_state(self, h):
        pass




