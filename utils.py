from abc import ABC, abstractmethod
from indicators import ZigZag, zz_opt
import numpy as np
from loguru import logger

class Position:
    def __init__(self, price, date, indx):
        self.open_price = abs(price)
        self.open_date = date
        self.open_indx = indx
        self.direction = np.sign(price)
        self.close_price = None
        self.close_date = None
        self.profit = None
        logger.debug(f"open position {self.open_indx} {'BUY' if self.direction > 0 else 'SELL'}: {self.open_date}: {self.open_price}")
    
    @property
    def duration(self):
        return self.close_indx - self.open_indx
    
    def close(self, price, date, indx):
        self.close_price = abs(price)
        self.close_date = date
        self.close_indx = indx
        self.profit = (self.close_price - self.open_price)*self.direction/self.open_price*100
        logger.debug(f"close position {'BUY' if self.direction > 0 else 'SELL'}: {self.close_date}: {self.close_price}, profit: {self.profit:4.2f}")
        return self.profit
    
class Broker:
    def __init__(self):
        self.active_orders = []
        self.active_position = None
        self.positions = []
    
    def close_orders(self):
        self.active_orders = []
    
    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])
        
    def update(self, h):
        for i, order in enumerate(self.active_orders): 
            oprice = abs(order["price"])
            logger.debug(f"{h.index[-1]} check {order['type'].upper()} order {order['price']:5.2f} << low: {h.Low[-1]}, high: {h.High[-1]}")
            triggered_price = None
            if order["type"] == "market":
                triggered_price = h.Open[-1]*np.sign(order["price"])
            else:
                if (h.Open[-1] > oprice and h.Close[-2] < oprice) or (h.Open[-1] < oprice and h.Close[-2] > oprice):
                    triggered_price = h.Open[-1]*np.sign(order["price"])                
                elif h.High[-1] > oprice and h.Low[-1] < oprice:
                    triggered_price = order["price"]


            if triggered_price is not None:
                logger.debug(f"triggered order {order},  (low: {h.Low[-1]}, high: {h.High[-1]})")
                
                self.active_orders.pop(i)
                if self.active_position is None:
                    self.active_position = Position(triggered_price, h.index[-1], h.shape[0])
                else:
                    if self.active_position.direction*triggered_price < 0:
                        self.active_position.close(triggered_price, h.index[-1], h.shape[0])
                        closed_position = self.active_position
                        self.active_position = None
                        self.positions.append(closed_position)
                        return closed_position
                    else:
                        raise NotImplementedError()
        return None    
                
    
class CandleFig(ABC):
    def __init__(self, min_history_size):
        self.min_history_size = min_history_size
        self.body_length = None
        self.orders = []
        self.body_line = None
            
    @abstractmethod
    def get_body(self) -> None:
        self.body_length = None
        self.body_line = None
        
    def update(self, h, t):
        assert h.shape[0] > self.min_history_size
        try:
            self.get_body(h[:t])
        except Exception as ex:
            logger.error(f"get body at {t} failed")
            raise ex
    
    @property
    def lines(self):
        lines = [self.body_line]
        return lines


class DenseLine(CandleFig):
    def __init__(self, body_maxsize, trend_maxsize, n_intersections, target_length):
        self.body_maxsize = body_maxsize
        self.trend_maxsize = trend_maxsize  
        self.n_intersections = n_intersections
        super(DenseLine, self).__init__(self.body_maxsize + self.trend_maxsize)
        self.center_line = None     
        self.target_length = target_length 
    
    def check_line(self, h, line):
        n, tlast = 0, 0
        for t in range(self.body_maxsize):
            if max(h.Close[-t], h.Open[-t]) > line and min(h.Close[-t], h.Open[-t]) < line:
                n += 1
                tlast = t
        return n, tlast
    
    def get_body(self, h):
        self.line = None
        line = h.Close[-1]
        n, t = self.check_line(h, line)
        if n >= self.n_intersections:
            self.body_length = t - 1
            self.center_line = line
            self.body_line = [(h.index[-self.body_length], line), (h.index[-1], line)]
            return True
        return False
    
    def get_trend(self, h) -> bool:
        self.trend_line = [(), (h.index[-1], self.center_line)]
        tmin = -self.trend_maxsize + h.Low[-self.trend_maxsize:].argmin()
        tmax = -self.trend_maxsize + h.High[-self.trend_maxsize:].argmax()
        if h.High[tmax] - self.center_line > self.center_line - h.Low[tmin]:
            self.trend_line[0] = (h.index[tmax], h.High[tmax])
            self.trend_length = -tmax
        else:
            self.trend_line[0] = (h.index[tmin], h.Low[tmin])    
            self.trend_length = -tmin           
    
    def get_prediction(self, h):
        self.prediction = 1 if self.trend_line[1][1] > self.trend_line[0][1] else -1
    
    def get_target(self, h):
        self.target_length = int(self.body_length*self.target_length)
        ptrg = h.Close.values[self.target_length]
        self.target = (ptrg - h.Close.values[0])/h.Close.values[0]
        self.target_line = [(h.index[0], self.center_line), (h.index[self.target_length], ptrg)]
  
        
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
        super(Trend, self).__init__(self.body_maxsize + self.trend_maxsize)  
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
                self.body_line = [(x, y) for x, y in zip(dates[-i:], values[-i:])]
                # self.get_trend(h[:-self.body_length+2])
                self.formation_found = True
                self.wait_for_enter = 10
            else:
                return
            
        lprice = max(self.body_line[-2][1], self.body_line[-3][1]) if self.trend_type > 0 else None
        sprice = min(self.body_line[-2][1], self.body_line[-3][1]) if self.trend_type < 0 else None 
        logger.debug(f"{h.index[-1]} long level: {lprice}, short level: {sprice}, close: {h.Close[-1]}")
                  
        if lprice and h.Close[-1] > lprice:
            self.orders = [{"type": "market", "price":h.Close[-1]}, 
                           {"type": "stop", "price":-h.Close[-1]*(1+self.tp/100)}, 
                           {"type": "stop", "price":-h.Close[-1]*(1-self.sl/100)}]
            logger.debug(f"{h.index[-1]} send order {self.orders[0]}, tp: {self.orders[1]}, sl: {self.orders[2]}")
        elif sprice and h.Close[-1] < sprice:
            self.orders = [{"type": "market", "price":-h.Close[-1]}, 
                           {"type": "stop", "price":h.Close[-1]*(1-self.tp/100)}, 
                           {"type": "stop", "price":h.Close[-1]*(1+self.sl/100)}]
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
            
