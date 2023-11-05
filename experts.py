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


class ExtensionBase:
    def __init__(self, cfg, name):
         self.name = name + ":" + ",".join([f"{k}={v}" for k, v in cfg.items()])


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
            
        logger.debug(f"{h.Id.iloc[-1]} long: {self.lprice}, short: {self.sprice}, cancel: {self.cprice}, close: {h.Close.iloc[-2]}")
        
        self.order_dir = 0
        if self.lprice:
            if h.Open.iloc[-1] > self.lprice:
                self.order_dir = 1
            if self.cprice and h.Open.iloc[-1] < self.cprice:
                self.reset_state()
                return
        if self.sprice:
            if h.Open.iloc[-1] < self.sprice:
                self.order_dir = -1
            if self.cprice and h.Open.iloc[-1] > self.cprice:
                self.reset_state()
                return            
            
        if self.order_dir != 0:        
            tp, sl = self.stops_processor(self, h)
            self.orders = [Order(self.order_dir, Order.TYPE.MARKET, h.Id.iloc[-1], h.index[-1])]
            if tp:
                self.orders.append(Order(tp, Order.TYPE.LIMIT, h.Id.iloc[-1], h.index[-1]))
            if sl:
                self.orders.append(Order(sl, Order.TYPE.LIMIT, h.Id.iloc[-1], h.index[-1]))
            logger.debug(f"{h.index[-1]} send order {self.orders[0]}, " + 
                         f"tp: {self.orders[1] if len(self.orders)>1 else 'NO'}, " +
                         f"sl: {self.orders[2] if len(self.orders)>2 else 'NO'}")
        
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
            

class ClsTrend(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTrend, self).__init__(cfg, name="trend")
        self.zigzag = ZigZag()
        
    def __call__(self, common, h) -> bool:
        # ids, dates, values, types = ZigZag().update(h)
        ids, dates, values, types = self.zigzag.update(h)
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
            if (self.cfg.npairs <= 2 and flag2) or (self.cfg.npairs == 3 and flag2 and flag3):
                is_fig = True
                trend_type = types[-2]
                            
        if is_fig:
            # if trend_type<0:
            #     return False
            i = self.cfg.npairs*2 + 1
            common.lines = [(x, y) for x, y in zip(dates[-i:-1], values[-i:-1])]
            # self.get_trend(h[:-self.body_length+2])
            common.lprice = max(common.lines[-1][1], common.lines[-2][1]) if trend_type > 0 else None
            common.sprice = min(common.lines[-1][1], common.lines[-2][1]) if trend_type < 0 else None
            common.cprice = common.lines[-2][1]
        return is_fig


class ClsTriangleSimp(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTriangleSimp, self).__init__(cfg, name="trngl_simp")
        self.zigzag = ZigZag()
        
    def __call__(self, common, h) -> bool:
        ids, dates, values, types = self.zigzag.update(h)
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
            if (self.cfg.npairs <= 2 and flag2) or (self.cfg.npairs == 3 and flag2 and flag3):
                is_fig = True
                    
        if is_fig:
            i = self.cfg.npairs*2 + 1
            common.lines = [(x, y) for x, y in zip(dates[-i:-1], values[-i:-1])]
            common.lprice = max(common.lines[-1][1], common.lines[-2][1])
            common.sprice = min(common.lines[-1][1], common.lines[-2][1]) 
        return is_fig


class ClsTriangleComp(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTriangleComp, self).__init__(cfg, name="trngl_comp")
        self.zigzag = ZigZag()
        
    def __call__(self, common, h) -> bool:
        ids, dates, values, types = self.zigzag.update(h)
        # ids, dates, values, types = zz_opt(h[-self.body_maxsize:])
        is_fig = False
        types_filt, vals_filt, ids_ = [], [], []
        for i in range(2, len(ids)):
            cur_type = types[-i]
            cur_val = values[-i]
            if len(types_filt) < 2:
                types_filt.append(cur_type)
                vals_filt.append(cur_val)
                ids_.append(-i)
            else:
                if len(types_filt) == 2:
                    valmax, valmin = max(vals_filt), min(vals_filt)
                if types_filt[-1] == 1 and cur_type == -1:
                    if cur_val <= valmin:
                        valmin = cur_val
                        types_filt.append(cur_type)
                        vals_filt.append(cur_val)
                        ids_.append(-i)
                if types_filt[-1] == -1 and cur_type == 1:
                    if cur_val >= valmax:
                        valmax = cur_val
                        types_filt.append(cur_type)
                        vals_filt.append(cur_val)  
                        ids_.append(-i)

        if len(types_filt) >= self.cfg.npairs*2:
            is_fig = True
            logger.debug(f"Found figure p-types : {types_filt}") 
            logger.debug(f"Found figure p-values: {vals_filt}") 
                    
        if is_fig:
            i = self.cfg.npairs*2 + 1
            common.lines = [(x, y) for x, y in zip([dates[j] for j in ids_[-i:][::-1]], [values[j] for j in ids_[-i:][::-1]])]
            # common.lines = [(x, y) for x, y in zip(dates[-ids_[-1]:-1], values[-ids_[-1]:-1])]
            common.lprice = max(common.lines[-1][1], common.lines[-2][1])
            common.sprice = min(common.lines[-1][1], common.lines[-2][1]) 

        return is_fig


class StopsFixed(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsFixed, self).__init__(cfg, name="stops_fix")
        
    def __call__(self, common, h):
        tp = -common.order_dir*h.Open[-1]*(1+common.order_dir*self.cfg.tp/100)
        sl = -common.order_dir*h.Open[-1]*(1-common.order_dir*self.cfg.sl/100)
        return tp, sl
    

class StopsDynamic(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsDynamic, self).__init__(cfg, name="stops_dyn")
        
    def __call__(self, common, h):
        tp, sl = None, None
        if self.cfg.tp_active:
            tp = -common.order_dir*(h.Open[-1] + common.order_dir*abs(common.lines[0][1]-common.lines[-1][1]))
        if self.cfg.sl_active:
            if common.order_dir > 0:
                sl = min(common.lines[-2][1], common.lines[-3][1])
            if common.order_dir < 0:
                sl = max(common.lines[-2][1], common.lines[-3][1])
            sl *= -common.order_dir
        return tp, sl

        
class PyConfig():
    def test(self):
        from configs.default import config
        for k, v in config.items():
            v = v.test
            if type(v) is EasyDict and "func" in v.keys():
                params = EasyDict({pk: pv.test for pk, pv in v.params.items()})
                config[k].func = v.func(params)
            else:
                config[k] = v
        return config

    def optim(self):
        from configs.default import config
        for k, vlist in config.items():
            vlist_new = []
            for v in vlist.optim:
                if type(v) is EasyDict and "func" in v.keys():
                    v.params = {pk: pv.optim for pk, pv in v.params.items()}
                    # v.func = partial(v.func, cfg=params)
                    params_list = self.unroll_params(v.params)
                    vlist_new += [EasyDict(func=v.func(params)) for params in params_list]
                else:
                    vlist_new.append(v)
            config[k] = vlist_new
        return config
    
    @staticmethod
    def unroll_params(cfg):
        import itertools
        keys, values = zip(*cfg.items())
        return [EasyDict(zip(keys, v)) for v in itertools.product(*values)]


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