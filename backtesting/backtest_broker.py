from time import perf_counter
from typing import List, Optional

import numpy as np
from loguru import logger

from common.type import Side
from common.utils import date2str
from data_processing.dataloading import MovingWindow
from trade.utils import ORDER_TYPE, Order, Position


class Broker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active_orders: List[Order] = []
        self.active_position: Position = None
        self.positions: List[Position] = []
        self.orders = []
        self.best_profit = 0
        
        self.time = None
        self.hist_id = None
        self.open_price = None
        
        self.mw = MovingWindow(cfg)

    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])

    @property
    def profits_abs(self):
        return np.array([p.profit_abs for p in self.positions])

    @property
    def fees_abs(self):
        return np.array([p.fees_abs for p in self.positions])

    def trade_stream(self, callback):
        for self.hist_window, dt in self.mw():
            self.time = self.hist_window["Date"][-1]
            self.hist_id = self.hist_window["Id"][-1]
            self.open_price = self.hist_window["Open"][-1]
            
            closed_position = self.update()
            callback({"timestamp": self.time})
            closed_position_new = self.update()
            if closed_position is None:
                if closed_position_new is not None:
                    closed_position = closed_position_new
            elif closed_position_new is not None:
                raise ValueError("closed positions disagreement!")
            self.update_state()
            
        if self.active_position is not None:
            self.close_active_pos(price=self.open_price, 
                                  time=self.time,
                                  hist_id=self.hist_id)

    def close_orders(self, hist_id, i=None):
        if i is not None:
            self.active_orders[i].close(hist_id)
            self.orders.append(self.active_orders.pop(i))
        else:
            while len(self.active_orders):
                self.active_orders[0].close(hist_id)
                self.orders.append(self.active_orders.pop(0))

    def update_sl(self, sl):
        if self.active_position is not None and sl is not None:
            assert not (self.active_position.side == Side.BUY and self.open_price <= sl), "sl must be below price if side = BUY" 
            assert not (self.active_position.side == Side.SELL and self.open_price >= sl), "sl must be above price if side = SELL" 
            self.active_position.update_sl(sl, self.time)

    def set_active_orders(self, new_orders_list: List[Order]):
        if len(new_orders_list):
            self.close_orders(self.hist_id)
            self.active_orders = new_orders_list

    def update_state(self):
        t0 = perf_counter()
        closed_position = self.update()
        # self.set_active_orders(h, new_orders_list)
        # closed_position_new = self.update(h)
        # if closed_position is None:
        #     if closed_position_new is not None:
        #         closed_position = closed_position_new
        # elif closed_position_new is not None:
        #     raise ValueError("closed positions disagreement!")
        # self.trailing_stop(h)
        return closed_position, perf_counter() - t0

    def close_active_pos(self, price, time, hist_id):
        self.active_position.close(price, time, hist_id)
        closed_position = self.active_position
        self.active_position = None
        self.positions.append(closed_position)
        return closed_position

    def update(self):
        closed_position = None
        last_low = self.hist_window["Low"][-2]
        last_high = self.hist_window["High"][-2]
        last_time = self.hist_window["Date"][-2]
        last_hist_id = self.hist_window["Id"][-2]
        
        if self.active_position is not None and self.active_position.sl is not None:
            self.active_orders.append(Order(
                            price=self.active_position.sl, 
                            side=Side.reverse(self.active_position.side), 
                            type=ORDER_TYPE.STOPLOSS,
                            volume=self.active_position.volume,
                            indx=self.active_position.open_indx, 
                            time=self.active_position.open_date))
            
        for i, order in enumerate(self.active_orders):
            triggered_price: float = None
            triggered_side: Optional[Side] = None
            triggered_date: np.datetime64 = None
            if order.type == ORDER_TYPE.MARKET and order.open_indx == self.hist_id:
                logger.info(
                    f"process order {order.id}"
                )
                triggered_price = self.open_price
                triggered_side = order.side
                triggered_date, triggered_id, triggered_vol = (
                    self.time,
                    self.hist_id,
                    order.volume,
                )
                # order.change(self.hist_id, self.open_price)
            if (order.type == ORDER_TYPE.LIMIT or order.type == ORDER_TYPE.STOPLOSS) and order.open_indx != self.hist_id:
                if (last_low > order.price and self.open_price < order.price) or (
                    last_high < order.price and self.open_price > order.price
                ):
                    logger.info(
                        f"process order {order.id}, and change price to O:{self.open_price}"
                    )
                    triggered_price = self.open_price
                    triggered_side = order.side
                    triggered_date, triggered_id, triggered_vol = (
                        self.time,
                        self.hist_id,
                        order.volume,
                    )
                elif last_high >= order.price and last_low <= order.price:
                    logger.info(
                        f"process order {order.id} (L:{last_low} <= {order.price:.2f} <= H:{last_high})"
                    )
                    triggered_price = order.price
                    triggered_side = order.side
                    triggered_date, triggered_id, triggered_vol = (
                        last_time,
                        last_hist_id,
                        order.volume,
                    )

            if triggered_price is not None:
                self.close_orders(triggered_id, i)

                if self.active_position is not None:
                    if self.active_position.side.value * triggered_side.value < 0:
                        closed_position = self.close_active_pos(triggered_price, triggered_date, triggered_id)
                        if order.type == ORDER_TYPE.LIMIT or order.type == ORDER_TYPE.STOPLOSS :
                            triggered_vol = 0
                    else:
                        raise NotImplementedError("Добор позиции не реализован")

                # Открытие новой позиции
                if triggered_vol:
                    sl = None
                    for order in self.active_orders:
                        if order.side.value * triggered_side.value < 0:
                            if (
                                triggered_side == Side.BUY and order.price < triggered_price
                            ) or (
                                triggered_side == Side.SELL and order.price > triggered_price
                            ):
                                sl = order.price

                    self.active_position = Position(
                        price=triggered_price,
                        side=triggered_side,
                        date=triggered_date,
                        indx=triggered_id,
                        ticker=self.cfg.ticker,
                        volume=triggered_vol,
                        period=self.cfg.period,
                        fee_rate=self.cfg.fee_rate,
                        sl=sl,
                    )
        for order in self.active_orders:
            if order.type == ORDER_TYPE.STOPLOSS:
                self.active_orders.remove(order)

        return closed_position

    def trailing_stop(self, h):
        date, p = self.hist_id, h.Close[-1]
        position = self.active_position
        if position is None or self.cfg.trailing_stop_rate == 0:
            return
        for order in self.active_orders:
            if date == order.open_date:
                self.best_profit = 0
            else:
                profit_cur = 0
                if self.active_position.side == Side.BUY:
                    profit_cur = h.High[-2] - self.active_position.open_price
                if self.active_position.side == Side.SELL:
                    profit_cur = self.active_position.open_price - h.Low[-2]
                if profit_cur >= self.best_profit:
                    self.best_profit = profit_cur

                order.change(
                    date,
                    order.price
                    + self.cfg.trailing_stop_rate * (self.open_price - order.price),
                )
