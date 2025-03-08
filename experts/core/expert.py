import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

# import torch
from backtesting.backtest_broker import Position
from common.type import Symbol
from common.utils import set_indicator_cache_dir
from experts.core.decision_maker import DecisionMaker
from experts.position_control import StopsController, TrailingStop


def init_target_from_cfg(cfg):
    cfg = deepcopy(cfg)
    Target = cfg.pop("type")
    return Target(cfg)

class ExpertBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        decision_maker_cfg = cfg["decision_maker"]
        decision_maker_cfg.update({k:cfg[k] for k in ["hist_buffer_size", "period", "symbol"]})
        self.decision_maker: DecisionMaker = init_target_from_cfg(decision_maker_cfg)
        assert "sl_processor" in cfg, "sl_processor must be defined in cfg"
        self.sl_processor: StopsController = init_target_from_cfg(cfg["sl_processor"])
        self.sl = None
        assert "tp_processor" in cfg, "tp_processor must be defined in cfg"
        self.tp_processor: StopsController = init_target_from_cfg(cfg.get("tp_processor", None))
        self.tp = None
        assert "trailing_stop" in cfg, "trailing_stop must be defined in cfg"
        self.trailing_stop: TrailingStop = init_target_from_cfg(cfg.get("trailing_stop", None))
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
        