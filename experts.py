from abc import ABC, abstractmethod
from indicators import ZigZag, zz_opt
import numpy as np
from loguru import logger
from utils import Order


class CandleFig(ABC):
    def __init__(self):
        self.lines = None
        self.orders = []
            
    @abstractmethod
    def get_body(self) -> None:
        self.lines = None
        
    def update(self, h):
        self.get_body(h)


class Triangle(CandleFig):
    def __init__(self, body_maxsize, n_pairs_of_tips, tp, sl):
        self.body_maxsize = body_maxsize
        self.trend_maxsize = 1  
        super(Triangle, self).__init__(self.body_maxsize + self.trend_maxsize)  
        self.npairs = n_pairs_of_tips
        self.tp = tp
        self.sl = sl
        self.formation_found = False
    
    def get_body(self, h):
        if self.formation_found == False:
            is_fig = False
            #ids, dates, values, types = ZigZag().update(h[-self.body_maxsize:])
            ids, dates, values, types = zz_opt(h[-self.body_maxsize:])
            
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
            
            # i = self.npairs*2 + 1
            # if len(ids) > 6:
            #     flag2, flag3 = False, False
            #     if types[-2] > 0:
            #         flag2 = values[-2] < values[-4] and values[-3] > values[-5]
            #         flag3 = values[-4] < values[-6] and values[-5] > values[-7]
            #     if types[-2] < 0:
            #         flag2 = values[-2] > values[-4] and values[-3] < values[-5]
            #         flag3 = values[-4] > values[-6]  and values[-5] < values[-7]
            #     if (self.npairs <= 2 and flag2) or (self.npairs == 3 and flag2 and flag3):
            #         is_fig = True
                        
            if is_fig:
                self.body_length = self.body_maxsize - ids[-i]+1
                self.body_line = [(x, y) for x, y in zip(dates[-i:], values[-i:])]
                # self.get_trend(h[:-self.body_length+2])
                self.formation_found = True

            else:
                return
            
        lprice = max(self.body_line[-2][1], self.body_line[-3][1])
        sprice = min(self.body_line[-2][1], self.body_line[-3][1])  
        logger.debug(f"long level: {lprice}, short level: {sprice}, close: {h.Close[-1]}")
                  
        if h.Close[-1] > lprice:
            self.orders = [{"type": "market", "price":h.Close[-1]}, 
                           {"type": "stop", "price":-h.Close[-1]*(1+self.tp/100)}, 
                           {"type": "stop", "price":-h.Close[-1]*(1-self.sl/100)}]
            logger.debug(f"send order {self.orders[0]} -> tp: {self.orders[1]}, sl: {self.orders[2]}")
        elif h.Close[-1] < sprice:
            self.orders = [{"type": "market", "price":-h.Close[-1]}, 
                           {"type": "stop", "price":h.Close[-1]*(1-self.tp/100)}, 
                           {"type": "stop", "price":h.Close[-1]*(1+self.sl/100)}]
            logger.debug(f"send order {self.orders[0]} -> tp: {self.orders[1]}, sl: {self.orders[2]}")
        else:
            pass
            # self.orders = [{"type": "stop", "price":lprice},
            #                 {"type": "stop", "price":-sprice},
            #                 {"type": "stop", "price":lprice-2},
            #                 {"type": "stop", "price":-sprice+2}]
            
    def get_trend(self, h):
        self.trend_length = 0
        body_tail = self.body_line[0][1]
        self.body_line = [(), (h.index[-1], body_tail)] + self.body_line
        tmin = -self.trend_maxsize + h.Low[-self.trend_maxsize:].argmin()
        tmax = -self.trend_maxsize + h.High[-self.trend_maxsize:].argmax()
        if h.High[tmax] - body_tail > body_tail - h.Low[tmin]:
            self.body_line[0] = (h.index[tmax], h.High[tmax])
            self.body_length += -tmax + 1
        else:
            self.body_line[0] = (h.index[tmin], h.Low[tmin])    
            self.body_length += -tmin + 1
            

class Trend(CandleFig):
    def __init__(self, body_maxsize, n_pairs_of_tips, tp, sl):
        self.body_maxsize = body_maxsize
        self.trend_maxsize = 1  
        super(Trend, self).__init__()  
        self.npairs = n_pairs_of_tips
        self.tp = tp
        self.sl = sl
        self.formation_found = False
        self.trend_type = 0
        self.wait_for_enter = 0
    
    def get_body(self, h):
        if self.formation_found == False:
            is_fig = False
            ids, dates, values, types = ZigZag().update(h[-self.body_maxsize:])
            #ids, dates, values, types = zz_opt(h[-self.body_maxsize:], self.npairs*2+2, simp_while_grow=False)
         
            i = self.npairs*2 + 1
            if len(ids) > 6:
                flag2, flag3 = False, False
                if types[-2] > 0:
                    flag2 = values[-2] > values[-4] and values[-3] > values[-5]
                    flag3 = values[-4] > values[-6] and values[-5] > values[-7]
                if types[-2] < 0:
                    flag2 = values[-2] < values[-4] and values[-3] < values[-5]
                    flag3 = values[-4] < values[-6] and values[-5] < values[-7]
                if (self.npairs <= 2 and flag2) or (self.npairs == 3 and flag2 and flag3):
                    is_fig = True
                    self.trend_type = types[-2]
                        
            if is_fig:
                self.body_length = self.body_maxsize - ids[-i]+1
                self.lines = [(x, y) for x, y in zip(dates[-i:], values[-i:])]
                # self.get_trend(h[:-self.body_length+2])
                self.formation_found = True
                self.wait_for_enter = 10
            else:
                return
            
        # lprice = max(self.lines[-2][1], self.lines[-3][1]) if self.trend_type > 0 else None
        # sprice = min(self.lines[-2][1], self.lines[-3][1]) if self.trend_type < 0 else None

        lprice = min(self.lines[-2][1], self.lines[-3][1]) if self.trend_type > 0 else None
        sprice = max(self.lines[-2][1], self.lines[-3][1]) if self.trend_type < 0 else None 
        
        logger.debug(f"{h.index[-1]} long level: {lprice}, short level: {sprice}, close: {h.Close[-2]}")
                  
        if lprice and h.Close[-2] >= lprice:
            self.orders = [Order(1, Order.TYPE.MARKET, h.Id[-1], h.index[-1]),
                           Order(-h.Close[-1]*(1+self.tp/100), Order.TYPE.LIMIT, h.Id[-1], h.index[-1]),
                           Order(-h.Close[-1]*(1-self.sl/100), Order.TYPE.LIMIT, h.Id[-1], h.index[-1])]
            logger.debug(f"{h.index[-1]} send order {self.orders[0]}, tp: {self.orders[1]}, sl: {self.orders[2]}")
        elif sprice and h.Close[-2] <= sprice:
            self.orders = [Order(-1, Order.TYPE.MARKET, h.Id[-1], h.index[-1]),
                           Order(h.Close[-1]*(1-self.tp/100), Order.TYPE.LIMIT, h.Id[-1], h.index[-1]),
                           Order(h.Close[-1]*(1+self.sl/100), Order.TYPE.LIMIT, h.Id[-1], h.index[-1])]
            logger.debug(f"{h.index[-1]} send order {self.orders[0]}, tp: {self.orders[1]}, sl: {self.orders[2]}")
        else:
            pass
            # self.orders = [{"type": "stop", "price":lprice},
            #                 {"type": "stop", "price":-sprice},
            #                 {"type": "stop", "price":lprice-2},
            #                 {"type": "stop", "price":-sprice+2}]
        
        if self.wait_for_enter == 0:
            self.formation_found = False
        else:
            self.wait_for_enter -= 1
            
            
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
            
