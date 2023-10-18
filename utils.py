from abc import ABC, abstractmethod
import numpy as np
from loguru import logger


class Order:
    class TYPE:
        MARKET = "market order"
        LIMIT = "limit order"
    
    def __init__(self, directed_price, type, indx, date):
        self.dir = np.sign(directed_price)
        self.price = abs(directed_price) if type == Order.TYPE.LIMIT else 0
        self.type = type
        self.indx = indx
        self.open_date = date
        self.close_date = None
        
    def __str__(self):
        return f"{self.type} {self.id}"
    
    @property
    def str_dir(self):
        return 'BUY' if self.dir > 0 else 'SELL'

    @property
    def id(self):
        return f"{self.str_dir}-{self.indx}-{self.price:.2f}"
    
    @property
    def lines(self):
        return [(self.open_date, self.price), (self.close_date, self.price)]

class Position:
    def __init__(self, price, date, indx):
        self.open_price = abs(price)
        self.open_date = date
        self.open_indx = indx
        self.dir = np.sign(price)
        self.close_price = None
        self.close_date = None
        self.profit = None
        logger.debug(f"{date} open position {self.id}")
    
    def __str__(self):
        return f"pos {self.dir} {self.id}"
    
    @property
    def str_dir(self):
        return 'BUY' if self.dir > 0 else 'SELL'
    
    @property
    def duration(self):
        return self.close_indx - self.open_indx
    
    @property
    def id(self):
        return f"{self.str_dir}-{self.open_indx}-{self.open_price:.2f}"
    
    def close(self, price, date, indx):
        self.close_price = abs(price)
        self.close_date = date
        self.close_indx = indx
        self.profit = (self.close_price - self.open_price)*self.dir/self.open_price*100
        logger.debug(f"{date} close position {self.id} at {self.close_price:.2f}, profit: {self.profit:.2f}")
        return self.profit
    
    @property
    def lines(self):
        return [(self.open_date, self.open_price), (self.close_date, self.close_price)]
    
class Broker:
    def __init__(self):
        self.active_orders = []
        self.active_position = None
        self.positions = []
        self.orders = []
    
    def close_orders(self, close_date, i=None):
        if i is not None:
            self.active_orders[i].close_date = close_date
            self.orders.append(self.active_orders.pop(i))
        else:            
            while len(self.active_orders):
                self.active_orders[0].close_date = close_date
                self.orders.append(self.active_orders.pop(0))
    
    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])
        
    def update(self, h):
        date = h.index[-1]
        for i, order in enumerate(self.active_orders): 
            triggered_price = None
            triggered_date = None
            if order.type == Order.TYPE.MARKET and order.indx == h.Id[-1]:
                logger.debug(f"{date} process order {order.id} (O:{h.Open[-1]})")
                triggered_price = h.Open[-1]*order.dir
                triggered_date = date
                order.price = h.Open[-1]
            if order.type == Order.TYPE.LIMIT and order.indx != h.Id[-1]:
                if (h.Low[-2] > order.price and h.Open[-1] < order.price) or (h.High[-2] < order.price and h.Open[-1] > order.price):
                    logger.debug(f"{date} process order {order.id}, and change price to O:{h.Open[-1]}")    
                    triggered_price = h.Open[-1]*order.dir 
                    triggered_date = date                        
                elif h.High[-2] >= order.price and h.Low[-2] <= order.price:
                    logger.debug(f"{date} process order {order.id} (L:{h.Low[-2]} <= {order.price:.2f} <= H:{h.High[-2]})")
                    triggered_price = order.price*order.dir
                    triggered_date = h.index[-2]
                    
            if triggered_price is not None:
                self.close_orders(triggered_date, i)
                if self.active_position is None:
                    self.active_position = Position(triggered_price, triggered_date, h.loc[triggered_date].Id)
                else:
                    if self.active_position.dir*triggered_price < 0:
                        self.active_position.close(triggered_price, triggered_date, h.loc[triggered_date].Id)
                        closed_position = self.active_position
                        self.active_position = None
                        self.positions.append(closed_position)
                        return closed_position
                    else:
                        raise NotImplementedError()
        return None    
                
    
