import numpy as np
from loguru import logger


class Order:
    class TYPE:
        MARKET = "market order"
        LIMIT = "limit order"
    
    def __init__(self, directed_price, type, indx, date):
        self.dir = np.sign(directed_price)
        self.type = type
        self.indx = indx
        self.open_date = date
        self.close_date = None
        self._change_hist = []
        self.change(date, abs(directed_price) if type == Order.TYPE.LIMIT else 0)

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
        # return [(self.open_date, self.price), (self.close_date, self.price)]
        return self._change_hist
    
    def change(self, date, price):
        self.price = price
        if price != 0:
            self._change_hist.append((date, price))
        
    def close(self, date):
        self.close_date = date
        self._change_hist.append((date, self.price))       
        

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.active_orders = []
        self.active_position = None
        self.positions = []
        self.orders = []
    
    def close_orders(self, close_date, i=None):
        if i is not None:
            self.active_orders[i].close(close_date)
            self.orders.append(self.active_orders.pop(i))
        else:            
            while len(self.active_orders):
                self.active_orders[0].close(close_date)
                self.orders.append(self.active_orders.pop(0))
    
    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])
        
    def update(self, h):
        date = h.index[-1]
        for i, order in enumerate(self.active_orders): 
            triggered_price = None
            triggered_date = None
            if order.type == Order.TYPE.MARKET and order.indx == h.Id.iloc[-1]:
                logger.debug(f"{date} process order {order.id} (O:{h.Open.iloc[-1]})")
                triggered_price = h.Open.iloc[-1]*order.dir
                triggered_date = date
                order.change(date, h.Open.iloc[-1])
            if order.type == Order.TYPE.LIMIT and order.indx != h.Id.iloc[-1]:
                if (h.Low.iloc[-2] > order.price and h.Open.iloc[-1] < order.price) or (h.High.iloc[-2] < order.price and h.Open.iloc[-1] > order.price):
                    logger.debug(f"{date} process order {order.id}, and change price to O:{h.Open.iloc[-1]}")    
                    triggered_price = h.Open.iloc[-1]*order.dir 
                    triggered_date = date                        
                elif h.High.iloc[-2] >= order.price and h.Low.iloc[-2] <= order.price:
                    logger.debug(f"{date} process order {order.id} (L:{h.Low.iloc[-2]} <= {order.price:.2f} <= H:{h.High.iloc[-2]})")
                    triggered_price = order.price*order.dir
                    triggered_date = h.index[-2]
                    
            if triggered_price is not None:
                self.close_orders(triggered_date, i)
                triggered_id = h.loc[triggered_date].Id
                if self.active_position is None:
                    self.active_position = Position(triggered_price, triggered_date, triggered_id)
                else:
                    if self.active_position.dir*triggered_price < 0:
                        self.active_position.close(triggered_price, triggered_date, triggered_id)
                        closed_position = self.active_position
                        self.active_position = None
                        self.positions.append(closed_position)
                        return closed_position
                    else:
                        # Докупка
                        pass
                        #raise NotImplementedError()
                    
        self.trailing_stop(h)
        return None    
                
    def trailing_stop(self, h):
        date, p = h.index[-1], h.Close.iloc[-1]
        position = self.active_position
        if position is None or self.cfg.trailing_stop_rate == 0 or 0:
            return
        for order in self.active_orders:
            if date == order.open_date:
                continue
            if position.dir == 1 and order.dir == -1 and p > order.price:
                order.change(date, order.price + self.cfg.trailing_stop_rate*(p - order.price))
            if position.dir == -1 and order.dir == 1 and p < order.price:
                order.change(date, order.price - self.cfg.trailing_stop_rate*(order.price - p))
        ss=1
                
                
def trailing_stop(orders:list[Order], position:Position, date, p, k):
    if position is None:
        return
    for order in orders:
        if position.dir == 1 and order.dir == -1 and p > order.price:
            order.change(date, order.price + k*(p - order.price))
        if position.dir == -1 and order.dir == 1 and p < order.price:
            order.change(date, order.price - k*(order.price - p))