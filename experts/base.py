from abc import ABC, abstractmethod
from time import perf_counter
from copy import deepcopy
import numpy as np
import yaml
from easydict import EasyDict
from loguru import logger
from pathlib import Path
import pandas as pd
from collections import OrderedDict

from indicators import *
from backtest_broker import Order
from data_processing.dataloading import build_features
# import torch
from backtest_broker import Broker, Position


class ExpertBase(ABC):
    def __init__(self):
        self.lines = []
        self.orders = []
            
    @abstractmethod
    def get_body(self) -> None:
        pass
    
    @abstractmethod
    def create_orders(self) -> None:
        pass
    
    def update(self, h, active_position: Position):
        t0 = perf_counter()
        self.active_position = active_position
        self.get_body(h)
        return perf_counter() - t0


class ExtensionBase:
    def __init__(self, cfg, name):
         self.name = name + ":" + "-".join([f"{v}" for k, v in cfg.items()])
         
    def __call__(self, common, h):
        pass
            
    def update_inner_state(self, h):
        pass


class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        self.cfg = cfg 
        super(ExpertFormation, self).__init__()  
        self.body_cls = cfg.body_classifier.func
        self.stops_processor = cfg.stops_processor.func
        self._reset_state()
        self.order_sent = False
        
        if self.cfg.run_model_device is not None:
            from ml import Net, Net2
            self.model = Net2(4, 32)
            self.model.load_state_dict(torch.load("model.pth"))
            # self.model.set_threshold(0.6)
            self.model.eval()
            self.model.to(self.cfg.run_model_device)
        
    def _reset_state(self):
        self.lprice = None
        self.sprice = None
        self.cprice = None
        self.formation_found = False
        self.order_dir = 0
            
    def estimate_volume(self, h):
        volume = self.cfg.wallet/h.Open[-1]*self.cfg.leverage
        volume = self.normalize_volume(volume)
        return volume
    
    def normalize_volume(self, volume):
        return round(volume/self.cfg.ticksize, 0)*self.cfg.ticksize
            
    def get_body(self, h):
        self.body_cls.update_inner_state(h)
        if not self.cfg.allow_overturn and self.active_position is not None:
            return
        
        self.order_sent = False
        self.order_dir = 0
        
        if self.cfg.allow_overturn or not self.formation_found:
            self.formation_found = self.body_cls(self, h)   
            
        logger.debug(f"{h.Id[-1]} long: {self.lprice}, short: {self.sprice}, cancel: {self.cprice}, open: {h.Open[-1]}")
        
        if self.lprice:
            if (self.sprice is None and h.Open[-1] >= self.lprice) or h.Close[-2] > self.lprice:
                self.order_dir = 1
            if self.cprice is not None and h.Open[-1] < self.cprice[1]:
                self._reset_state()
                return
            
        if self.sprice:
            if (self.lprice is None and h.Open[-1] <= self.sprice) or h.Close[-2] < self.sprice:
                self.order_dir = -1
            if self.cprice and h.Open[-1] > self.cprice[-1]:
                self._reset_state()
                return            
        
        if h.Date[-1] in self.cfg.no_trading_days:
            self._reset_state()
            
        # y = None
        # if self.cfg.run_model_device and self.order_dir != 0:
        #     x = build_features(h, 
        #                        self.order_dir, 
        #                        self.stops_processor.cfg.sl,
        #                        self.cfg.trailing_stop_rate
        #                        )
        #     x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(self.cfg.run_model_device)
        #     y = [0.5, 1, 2, 4, 8][self.model.predict(x).item()]
            
        
        if self.order_dir != 0:
            if self.active_position is None or self.active_position.dir*self.order_dir < 0:
                tp, sl = self.stops_processor(self, h)
                self.create_orders(h.Id[-1], self.order_dir, self.estimate_volume(h), tp, sl)
                self.order_sent = True
                if not self.cfg.allow_overturn:
                    self._reset_state()

            
class BacktestExpert(ExpertFormation):
    def __init__(self, cfg):
        self.cfg = cfg
        super(BacktestExpert, self).__init__(cfg)
        
    def create_orders(self, time_id, dir, volume, tp, sl):
        self.orders = [Order(dir, Order.TYPE.MARKET, volume, time_id, time_id)]
        log_message = f"{time_id} send order {self.orders[0]}"
        if sl:
            self.orders.append(Order(sl, Order.TYPE.LIMIT, volume, time_id, time_id))
            log_message += f", sl: {self.orders[-1]}"     
        if tp:
            self.orders.append(Order(tp, Order.TYPE.LIMIT, volume, time_id, time_id))
            log_message += f", tp: {self.orders[-1]}"

        logger.debug(log_message)
        
            
class ByBitExpert(ExpertFormation):
    def __init__(self, cfg, session):
        self.cfg = cfg
        self.session = session
        super(ByBitExpert, self).__init__(cfg)
        
    def create_orders(self, time_id, order_dir, volume, tp, sl):
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=self.cfg.ticker,
                side="Buy" if order_dir > 0 else "Sell",
                orderType="Market",
                qty=str(volume),
                timeInForce="GTC",
                # orderLinkId="spot-test-postonly",
                stopLoss="" if sl is None else str(abs(sl)),
                takeProfit="" if tp is None else str(tp)
                )
            logger.debug(resp)
        except Exception as ex:
            logger.error(ex)


class StopsFixed(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsFixed, self).__init__(cfg, name="stops_fix")
        
    def __call__(self, common, h, sl_custom=None):
        tp = -common.order_dir*h.Open[-1]*(1+common.order_dir*self.cfg.tp*self.cfg.sl/100) if self.cfg.tp is not None else self.cfg.tp
        sl = self.cfg.sl
        if sl_custom is not None:
            sl = sl_custom
        if sl is not None:
            sl = -common.order_dir*h.Open[-1]*(1-common.order_dir*sl/100)
            
        return tp, sl
    

class StopsDynamic(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsDynamic, self).__init__(cfg, name="stops_dyn")
        
    def __call__(self, common, h):
        tp, sl = None, None
        if self.cfg.tp_active:
            tp = -common.order_dir*common.tp[common.order_dir]
        if self.cfg.sl_active:
            sl_add = 0#(common.sl[-1] - common.sl[1])
            sl = -common.order_dir*(common.sl[common.order_dir] - common.order_dir*sl_add)
        return tp, sl
