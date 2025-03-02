from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from common.type import Side
from data_processing.dataloading import MovingWindow
from trade.utils import ORDER_TYPE, Order, Position

from .profiling import line_profile_function, profile_function


class Broker:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg
        self.active_orders: List[Order] = []
        self.active_position: Position = None
        self.positions: List[Position] = []
        self.orders = []
        self.profit_hist = {"dates": [], "profit_nofees": [], "fees_csum": []}
        
        self.time = None
        self.hist_id = None
        self.open_price = None
        
        if cfg is not None:
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

    @profile_function
    def trade_stream(self, callback):
        for self.hist_window, dt in tqdm(self.mw(), desc="Backtest", total=self.mw.timesteps_count, disable=False):
            self.time = self.hist_window["Date"][-1]
            self.hist_id = self.hist_window["Id"][-1]
            self.open_price = self.hist_window["Open"][-1]
            
            closed_position = self.update()
            callback({"timestamp": self.time})  # TODO remove time
            closed_position_new = self.update()
            if closed_position is None:
                if closed_position_new is not None:
                    closed_position = closed_position_new
            elif closed_position_new is not None:
                if self.update() is not None:
                    raise ValueError("closed positions disagreement!")
            self.update_profit_curve()
            
        if self.active_position is not None:
            self.close_active_pos(price=self.open_price, 
                                  time=self.time,
                                  hist_id=self.hist_id)
            self.update_profit_curve()
        self.profit_hist = pd.DataFrame(self.profit_hist)

    def update_profit_curve(self):
        """
        Update cumulative profit curve from current state and active position.
        This method computes the cumulative profit from all closed positions and adds
        the unrealized profit from the active position (if any) using the current open price.
        It also updates the best profit record if the current cumulative profit exceeds it.
        """
        # Cumulative profit from closed positions
        cumulative_profit = self.profits_abs.sum()
        
        # If an active position exists, add its unrealized profit
        if self.active_position is not None:
            if self.active_position.side == Side.BUY:
                active_profit = (self.open_price - self.active_position.open_price) * self.active_position.volume
            elif self.active_position.side == Side.SELL:
                active_profit = (self.active_position.open_price - self.open_price) * self.active_position.volume
            else:
                active_profit = 0
            cumulative_profit += active_profit
        
        # Append current cumulative profit to the profit curve
        self.profit_hist["dates"].append(pd.to_datetime(self.time))        
        self.profit_hist["profit_nofees"].append(cumulative_profit)
        self.profit_hist["fees_csum"].append(self.fees_abs.sum())

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
            self.active_position.update_sl(sl, self.time)

    def update_tp(self, tp):
        if self.active_position is not None and tp is not None:
            self.active_position.update_tp(tp, self.time)

    def set_active_orders(self, new_orders_list: List[Order]):
        if len(new_orders_list):
            self.close_orders(self.hist_id)
            self.active_orders = new_orders_list

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

        if self.active_position is not None and self.active_position.tp is not None:
            self.active_orders.append(Order(
                            price=self.active_position.tp, 
                            side=Side.reverse(self.active_position.side), 
                            type=ORDER_TYPE.TAKEPROF,
                            volume=self.active_position.volume,
                            indx=self.active_position.open_indx, 
                            time=self.active_position.open_date))
            
        for i, order in enumerate(self.active_orders):
            triggered_price: float = None
            triggered_side: Optional[Side] = None
            triggered_date: np.datetime64 = None
            if order.type == ORDER_TYPE.MARKET and order.open_indx == self.hist_id:
                logger.debug(
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
            if order.type in (ORDER_TYPE.LIMIT, ORDER_TYPE.STOPLOSS, ORDER_TYPE.TAKEPROF) and order.open_indx != self.hist_id:
                if (last_low > order.price and self.open_price < order.price) or (
                    last_high < order.price and self.open_price > order.price
                ):
                    logger.debug(
                        f"process order {order.id}, and set price to {self.open_price}"
                    )
                    triggered_price = self.open_price
                    triggered_side = order.side
                    triggered_date, triggered_id, triggered_vol = (
                        self.time,
                        self.hist_id,
                        order.volume,
                    )
                elif last_high >= order.price and last_low <= order.price:
                    logger.debug(
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
                        if triggered_vol >= self.active_position.volume:
                            # If triggered volume is greater or equal to position volume,
                            # close entire position and reduce triggered volume
                            closed_position = self.close_active_pos(triggered_price, triggered_date, triggered_id)
                            triggered_vol -= closed_position.volume
                        else:
                            self.active_position.trim_position(triggered_vol, triggered_price, triggered_date)
                            triggered_vol = 0
                    else:
                        # Add volume to the existing position
                        self.active_position.add_to_position(triggered_vol, triggered_price, triggered_date)
                        triggered_vol = 0

                # Open new position if there is remaining volume
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
                        ticker=self.cfg["symbol"].ticker,
                        volume=triggered_vol,
                        period=self.cfg['period'],
                        fee_rate=self.cfg['fee_rate'],
                        sl=sl,
                    )
                    triggered_vol = 0
        for order in self.active_orders:
            if order.type in (ORDER_TYPE.STOPLOSS, ORDER_TYPE.TAKEPROF):
                self.active_orders.remove(order)

        return closed_position