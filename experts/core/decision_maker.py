from abc import ABC, abstractmethod
import pandas as pd
from common.type import Line, Side


class DecisionMaker(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sl_definer = {Side.BUY: None, Side.SELL: None}
        self.tp_definer = {Side.BUY: None, Side.SELL: None}
        self.lprice, self.sprice, self.cprice, self.tsignal = None, None, None, None
        self.indicator_vis_objects = None
        self.vis_items = []
        self.target_volume_fraction = None
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
    def look_around(self, h) -> bool:
        pass
    
    @abstractmethod
    def update_inner_state(self, h):
        pass

    


