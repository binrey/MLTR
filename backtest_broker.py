from dataclasses import dataclass
import numpy as np
from loguru import logger
from time import perf_counter
from typing import List
from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1
    UNDEF = 0


def date2str(date):
    return str(date).split('.')[0]


class Order:
    class TYPE:
        MARKET = "market order"
        LIMIT = "limit order"
    
    def __init__(self, directed_price, type, volume, indx, date):
        self.dir = np.sign(directed_price)
        self.type = type
        self.volume = volume
        self.open_indx = indx
        self.open_date = date
        self.close_date = None
        self._change_hist = []
        self.change(date, abs(directed_price) if type == Order.TYPE.LIMIT else 0)

    def __str__(self):
        return f"{self.type} {self.id} {self.volume}"
    
    @property
    def str_dir(self):
        return 'BUY' if self.dir > 0 else 'SELL'

    @property
    def id(self):
        return f"{self.open_indx}-{self.str_dir}-{self.price:.2f}"
    
    @property
    def lines(self):
        return self._change_hist
    
    def change(self, date, price):
        assert round(date) == date
        self.price = price
        if price != 0:
            self._change_hist.append((date, price))
        
    def close(self, date):
        assert round(date) == date
        self.close_date = date
        self._change_hist.append((date, self.price))       
        

class Position:
    def __init__(self, price, date, indx, ticker="NoName", volume=1, period="M5", sl=None, fee_rate=0):
        self.volume = float(volume)
        assert self.volume > 0
        self.ticker = ticker
        self.period = period
        self.open_price = abs(float(price))
        self.open_date = np.datetime64(date)
        self.open_indx = int(indx)
        self.sl = float(sl) if sl is not None else sl
        self.open_risk = np.nan
        if self.sl is not None:
            self.open_risk = abs(self.open_price - self.sl)/self.open_price*100
        self.dir = np.sign(float(price))
        self.close_price = None
        self.close_date = None
        self.profit = None
        self.profit_abs = None
        self.fee_rate  = fee_rate
        self.fees = 0
        self.fees_abs = 0
        self._update_fees(self.open_price, self.volume)
        logger.debug(f"{date2str(date)} open position {self.id}")
    
    def __str__(self):
        return f"pos {self.ticker} {self.dir} {self.volume} {self.id}"
    
    def order_fees(self, price, volume):
        return price*volume*self.fee_rate/100
    
    def _update_fees(self, price, volume):
        self.fees_abs += self.order_fees(price, volume)
        self.fees = self.fees_abs/self.volume/self.open_price*100
    
    @property
    def str_dir(self):
        return 'BUY' if self.dir > 0 else 'SELL'
    
    @property
    def duration(self):
        return self.close_indx - self.open_indx
    
    @property
    def id(self):
        return f"{self.open_indx}-{self.str_dir}-{self.open_price:.2f}-{self.volume:.2f}"
    
    def close(self, price, date, indx):
        self.close_price = abs(price)
        self.close_date = np.datetime64(date)
        self.close_indx = indx
        self._update_fees(self.close_price, self.volume)
        self.profit_abs = (self.close_price - self.open_price)*self.dir*self.volume
        self.profit = self.profit_abs/self.open_price*100 - self.fees
        self.profit_abs -= self.fees_abs
        
        logger.debug(f"{date2str(date)} close position {self.id} at {self.close_price:.2f}, profit: {self.profit_abs:.2f} ({self.profit:.2f}%)")
    
    @property
    def lines(self):
        return [(self.open_indx, self.open_price), (self.close_indx, self.close_price)]
    
class Broker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active_orders = []
        self.active_position = None
        self.positions: List[Position] = []
        self.orders = []
        
        self.best_profit = 0
        self.correction = 0
        self.add_profit = 0
    
    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])
    
    @property
    def profits_abs(self):
        return np.array([p.profit_abs for p in self.positions])    

    @property
    def fees(self):
        return np.array([p.fees_abs for p in self.positions])

    def close_orders(self, close_date, i=None):
        if i is not None:
            self.active_orders[i].close(close_date)
            self.orders.append(self.active_orders.pop(i))
        else:            
            while len(self.active_orders):
                self.active_orders[0].close(close_date)
                self.orders.append(self.active_orders.pop(0))
                
    def set_active_orders(self, h, new_orders_list: List[Order]):
        if len(new_orders_list):
            self.close_orders(h.Id[-1])
            self.active_orders = new_orders_list
    
    def update_state(self, h, new_orders_list: List[Order]):
        t0 = perf_counter()
        closed_position = self.update(h)
        self.set_active_orders(h, new_orders_list)
        closed_position_new = self.update(h)
        if closed_position is None:
            if closed_position_new is not None:
                closed_position = closed_position_new
        elif closed_position_new is not None:
            raise ValueError("closed positions disagreement!")
        self.trailing_stop(h)
        return closed_position, perf_counter() - t0
        
    def update(self, h):
        date = h.Date[-1]
        closed_position = None
        for i, order in enumerate(self.active_orders): 
            triggered_price = None
            triggered_date = None
            if order.type == Order.TYPE.MARKET and order.open_indx == h.Id[-1]:
                logger.debug(f"{date2str(date)} process order {order.id} (O:{h.Open[-1]})")
                triggered_price = h.Open[-1]*order.dir
                triggered_date, triggered_id, triggered_vol = date, h.Id[-1], order.volume
                order.change(h.Id[-1], h.Open[-1])
            if order.type == Order.TYPE.LIMIT and order.open_indx != h.Id[-1]:
                if (h.Low[-2] > order.price and h.Open[-1] < order.price) or (h.High[-2] < order.price and h.Open[-1] > order.price):
                    logger.debug(f"{date2str(date)} process order {order.id}, and change price to O:{h.Open[-1]}")    
                    triggered_price = h.Open[-1]*order.dir 
                    triggered_date, triggered_id, triggered_vol = date, h.Id[-1], order.volume                     
                elif h.High[-2] >= order.price and h.Low[-2] <= order.price:
                    logger.debug(f"{date2str(date)} process order {order.id} (L:{h.Low[-2]} <= {order.price:.2f} <= H:{h.High[-2]})")
                    triggered_price = order.price*order.dir
                    triggered_date, triggered_id, triggered_vol = h.Date[-2], h.Id[-2], order.volume
                    
            if triggered_price is not None:
                self.close_orders(triggered_id, i)

                if self.active_position is not None:
                    if self.active_position.dir*triggered_price < 0:
                        self.active_position.close(triggered_price, triggered_date, triggered_id)
                        closed_position = self.active_position
                        self.active_position = None
                        self.positions.append(closed_position)
                        if order.type == Order.TYPE.LIMIT:
                            triggered_vol = 0
                    else:
                        # Добор позиции
                        raise NotImplementedError()
                
                # Открытие новой позиции 
                if triggered_vol: 
                    sl = None
                    for order in self.active_orders:
                        if order.dir*triggered_price < 0:
                            if (triggered_price > 0 and order.price < triggered_price) or (triggered_price < 0 and order.price > abs(triggered_price)):
                                sl = order.price
                
                    self.active_position = Position(price=triggered_price, 
                                                    date=triggered_date, 
                                                    indx=triggered_id, 
                                                    ticker=self.cfg.ticker,
                                                    volume=triggered_vol,
                                                    period=self.cfg.period, 
                                                    fee_rate=self.cfg.fee_rate,
                                                    sl=sl)

        return closed_position
                   
    def trailing_stop(self, h):
        date, p = h.Id[-1], h.Close[-1]
        position = self.active_position
        if position is None or self.cfg.trailing_stop_rate == 0:
            return
        for order in self.active_orders:
            if date == order.open_date:
                self.best_profit = 0
                self.correction = 0
                self.max_correction = 0
                self.add_profit = 0
                continue
            t = h.Id[-1] - order.open_indx
            sl_rate = self.cfg.trailing_stop_rate
            dp = h.Open[-1] - order.price
            order.change(date, order.price + sl_rate * dp)