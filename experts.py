from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import yaml
from easydict import EasyDict
from loguru import logger

from indicators import ZigZag, zz_opt
from utils import Order


class ExpertBase(ABC):
    def __init__(self):
        self.lines = None
        self.orders = []
            
    @abstractmethod
    def get_body(self) -> None:
        pass
        
    def update(self, h):
        self.get_body(h)


class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        self.trend_maxsize = 1  
        super(ExpertFormation, self).__init__()  
        self.body_cls = cfg.body_classifier.func
        self.stops_processor = cfg.stops_processor.func
        self.wait_length = cfg.wait_entry_point
        self.reset_state()
        
    def reset_state(self):
        self.formation_found = False
        self.wait_entry_point = 0
        self.lprice = None
        self.sprice = None
        self.cprice = None
            
    def get_body(self, h):
        if self.formation_found == False:
            self.formation_found = self.body_cls(self, h)
            if self.formation_found:
                self.wait_entry_point = self.wait_length
            else:
                return
            
        logger.debug(f"{h.index[-1]} long: {self.lprice}, short: {self.sprice}, cancel: {self.sprice}, close: {h.Close[-2]}")
        
        self.order_dir = 0
        if self.lprice:
            if h.Open[-1] > self.lprice:
                self.order_dir = 1
            if h.Open[-1] < self.cprice:
                self.reset_state()
                return
        elif self.sprice:
            if h.Open[-1] < self.sprice:
                self.order_dir = -1
            if h.Open[-1] > self.cprice:
                self.reset_state()
                return            
            
        if self.order_dir != 0:        
            tp, sl = self.stops_processor(self, h)
            self.orders = [Order(self.order_dir, Order.TYPE.MARKET, h.Id[-1], h.index[-1]),
                           Order(tp, Order.TYPE.LIMIT, h.Id[-1], h.index[-1]),
                           Order(sl, Order.TYPE.LIMIT, h.Id[-1], h.index[-1])]
            logger.debug(f"{h.index[-1]} send order {self.orders[0]}, tp: {self.orders[1]}, sl: {self.orders[2]}")
        
        if self.wait_entry_point == 0:
            self.formation_found = False
        else:
            self.wait_entry_point -= 1
            
            
    def get_trend(self, h):
        self.trend_length = 0
        body_tail = self.lines[0][1]
        self.lines = [(), (h.index[-1], body_tail)] + self.lines
        tmin = -self.trend_maxsize + h.Low[-self.trend_maxsize:].argmin()
        tmax = -self.trend_maxsize + h.High[-self.trend_maxsize:].argmax()
        if h.High[tmax] - body_tail > body_tail - h.Low[tmin]:
            self.lines[0] = (h.index[tmax], h.High[tmax])
            self.body_length += -tmax + 1
        else:
            self.lines[0] = (h.index[tmin], h.Low[tmin])    
            self.body_length += -tmin + 1
            

def cls_trend(self, h, cfg) -> bool:
    # cfg = self.cfg.body_classifier
    ids, dates, values, types = ZigZag().update(h)
    #ids, dates, values, types = zz_opt(h, self.npairs*2+2, simp_while_grow=False)
    is_fig = False
    if len(ids) > 6:
        flag2, flag3 = False, False
        if types[-2] > 0:
            flag2 = values[-2] > values[-4] and values[-3] > values[-5]
            flag3 = values[-4] > values[-6] and values[-5] > values[-7]
        if types[-2] < 0:
            flag2 = values[-2] < values[-4] and values[-3] < values[-5]
            flag3 = values[-4] < values[-6] and values[-5] < values[-7]
        if (cfg.npairs <= 2 and flag2) or (cfg.npairs == 3 and flag2 and flag3):
            is_fig = True
            trend_type = types[-2]
                
    if is_fig:
        i = cfg.npairs*2 + 1
        self.lines = [(x, y) for x, y in zip(dates[-i:-1], values[-i:-1])]
        # self.get_trend(h[:-self.body_length+2])
        self.lprice = max(self.lines[-1][1], self.lines[-2][1]) if trend_type > 0 else None
        self.sprice = min(self.lines[-1][1], self.lines[-2][1]) if trend_type < 0 else None
        self.cprice = self.lines[-2][1]
    return is_fig


def cls_triangle_simple(self, h, cfg) -> bool:
    ids, dates, values, types = ZigZag().update(h)
    # ids, dates, values, types = zz_opt(h, self.npairs*2+2, simp_while_grow=False)
    is_fig = False
    if len(ids) > 6:
        flag2, flag3 = False, False
        if types[-2] > 0:
            flag2 = values[-2] < values[-4] and values[-3] > values[-5]
            flag3 = values[-4] < values[-6] and values[-5] > values[-7]
        if types[-2] < 0:
            flag2 = values[-2] > values[-4] and values[-3] < values[-5]
            flag3 = values[-4] > values[-6]  and values[-5] < values[-7]
        if (cfg.npairs <= 2 and flag2) or (cfg.npairs == 3 and flag2 and flag3):
            is_fig = True
                
    if is_fig:
        i = cfg.npairs*2 + 1
        self.lines = [(x, y) for x, y in zip(dates[-i:-1], values[-i:-1])]
        # self.get_trend(h[:-self.body_length+2])
        self.lprice = max(self.lines[-1][1], self.lines[-2][1])
        self.sprice = min(self.lines[-1][1], self.lines[-2][1]) 
        self.tp = abs(values[0] - values[1])

    return is_fig


def cls_triangle_complex(self, h, cfg):
    ids, dates, values, types = ZigZag().update(h[-self.body_maxsize:])
    # ids, dates, values, types = zz_opt(h[-self.body_maxsize:])
    
    types_filt, vals_filt = [], []
    for i in range(2, len(ids)):
        cur_type = types[-i]
        cur_val = values[-i]
        if len(types_filt) < 2:
            types_filt.append(cur_type)
            vals_filt.append(cur_val)
        else:
            if len(types_filt) == 2:
                valmax, valmin = max(vals_filt), min(vals_filt)
            if types_filt[-1] == 1 and cur_type == -1:
                if cur_val <= valmin:
                    valmin = cur_val
                    types_filt.append(cur_type)
                    vals_filt.append(cur_val)
            if types_filt[-1] == -1 and cur_type == 1:
                if cur_val >= valmax:
                    valmax = cur_val
                    types_filt.append(cur_type)
                    vals_filt.append(cur_val)  

    if len(types_filt) >= self.npairs*2:
        is_fig = True
        logger.debug(f"Found figure p-types : {types_filt}") 
        logger.debug(f"Found figure p-values: {vals_filt}") 
                
    if is_fig:
        self.lines = [(x, y) for x, y in zip(dates[-i:], values[-i:])]
        # self.get_trend(h[:-self.body_length+2])
        lprice = max(self.lines[-2][1], self.lines[-3][1])
        sprice = min(self.lines[-2][1], self.lines[-3][1]) 

    return is_fig, lprice, sprice


def stops_fixed(self, h, cfg):
    tp = -self.order_dir*h.Open[-1]*(1+self.order_dir*cfg.tp/100)
    sl = -self.order_dir*h.Open[-1]*(1-self.order_dir*cfg.sl/100)
    return tp, sl
    

def stops_dynamic(self, h, cfg):
    tp = -self.order_dir*(h.Open[-1] + self.order_dir*abs(self.lines[-1][1]-self.lines[-4][1]))
    sl = -self.order_dir*self.lines[-2][1]#(h.Open[-1] - abs(self.lines[-2][1]-self.lines[-3][1]))
    return tp, sl

class YAMLConfigReader:
    func_lib = {"cls_trend": cls_trend,
                "cls_trngl_simp": cls_triangle_simple}
    
    def __init__(self):
        pass
        
    def read(self, cfg):
        cfg = Config(yaml.safe_load(open(cfg, "r")))
        ftype = cfg.body_classifier.type
        if ftype in self.func_lib.keys():
            cfg.body_classifier["func"] = partial(self.func_lib[ftype], cfg=Config(cfg.body_classifier.copy()))
        else:
            logger.error(f"{ftype} wrong body_classifier type")
        logger.info(cfg)
        return cfg

        
def pyconfig():
    from configs.default import test_config
    for k, v in test_config.items():
        if type(v) is EasyDict and "func" in v.keys():
            v.func = partial(v.func, cfg=v.params)
    return test_config


class Config(EasyDict):
    def __str__(self):
        out = "config file:\n"
        for k, v in self.__dict__.items():
            if type(v) is Config:
                out += f"{k}:\n"
                for kk, vv in v.items():
                    out += f"  {kk}: {vv}\n"
            else:
                out += f"{k}: {v}\n"
        return out