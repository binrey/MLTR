from time import perf_counter
from typing import List

import numpy as np
from loguru import logger

from trade.utils import Order, Position
from type import Side
from utils import date2str


class Broker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active_orders = []
        self.active_position = None
        self.positions: List[Position] = []
        self.orders = []
        self.best_profit = 0

    @property
    def profits(self):
        return np.array([p.profit for p in self.positions])

    @property
    def profits_abs(self):
        return np.array([p.profit_abs for p in self.positions])

    @property
    def fees_abs(self):
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
                logger.debug(
                    f"{date2str(date)} process order {order.id} (O:{h.Open[-1]})"
                )
                triggered_price = h.Open[-1] * order.side.value
                triggered_date, triggered_id, triggered_vol = (
                    date,
                    h.Id[-1],
                    order.volume,
                )
                order.change(h.Id[-1], h.Open[-1])
            if order.type == Order.TYPE.LIMIT and order.open_indx != h.Id[-1]:
                if (h.Low[-2] > order.price and h.Open[-1] < order.price) or (
                    h.High[-2] < order.price and h.Open[-1] > order.price
                ):
                    logger.debug(
                        f"{date2str(date)} process order {order.id}, and change price to O:{h.Open[-1]}"
                    )
                    triggered_price = h.Open[-1] * order.side.value
                    triggered_date, triggered_id, triggered_vol = (
                        date,
                        h.Id[-1],
                        order.volume,
                    )
                elif h.High[-2] >= order.price and h.Low[-2] <= order.price:
                    logger.debug(
                        f"{date2str(date)} process order {order.id} (L:{h.Low[-2]} <= {order.price:.2f} <= H:{h.High[-2]})"
                    )
                    triggered_price = order.price * order.side.value
                    triggered_date, triggered_id, triggered_vol = (
                        h.Date[-2],
                        h.Id[-2],
                        order.volume,
                    )

            if triggered_price is not None:
                self.close_orders(triggered_id, i)

                if self.active_position is not None:
                    if self.active_position.side.value * triggered_price < 0:
                        self.active_position.close(
                            triggered_price, triggered_date, triggered_id
                        )
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
                        if order.side.value * triggered_price < 0:
                            if (
                                triggered_price > 0 and order.price < triggered_price
                            ) or (
                                triggered_price < 0
                                and order.price > abs(triggered_price)
                            ):
                                sl = order.price

                    self.active_position = Position(
                        price=triggered_price,
                        date=triggered_date,
                        indx=triggered_id,
                        ticker=self.cfg.ticker,
                        volume=triggered_vol,
                        period=self.cfg.period,
                        fee_rate=self.cfg.fee_rate,
                        sl=sl,
                    )

        return closed_position

    def trailing_stop(self, h):
        date, p = h.Id[-1], h.Close[-1]
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

                # dp = self.active_position.open_price + self.active_position.dir * self.best_profit
                # new_sl = dp * (1 - self.active_position.dir*self.cfg.trailing_stop_rate)
                # if (new_sl - order.price) * self.active_position.dir >= 0:
                #     order.change(date, new_sl)

                order.change(
                    date,
                    order.price
                    + self.cfg.trailing_stop_rate * (h.Open[-1] - order.price),
                )
