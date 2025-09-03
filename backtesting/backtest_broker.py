from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from common.type import Side, Symbol, VolEstimRule, VolumeControl, to_datetime
from data_processing.dataloading import MovingWindow
from trade.utils import ORDER_TYPE, Order, Position


class TradeHistory:
    def __init__(self, moving_window: MovingWindow, positions: List[Position]):
        self.mw = moving_window
        self.mw.size = 1
        self.profit_hist = defaultdict(list)
        self.leverage = 1
        self.positions = positions

        self.posdict_open: Dict[np.datetime64, int] = {
            self.mw.period.round_to_period(pos.open_date): i for i, pos in enumerate(positions)}
        self.posdict_closed: Dict[np.datetime64, int] = {
            self.mw.period.round_to_period(pos.close_date): i for i, pos in enumerate(positions)}
        self.cumulative_profit = 0
        self.cumulative_fees = 0
        active_position, max_profit = None, 0
        self.deposit, self.max_loss = None, None
        self.wallet = 0
        self.volume_control = None
        self.date_start = self.mw.date_start
        self.date_end = self.mw.date_end

        for self.hist_window, _ in tqdm(self.mw(), desc="Build profit curve", total=self.mw.timesteps_count, disable=True):
            cur_time = self.hist_window["Date"][-1]
            closed_position_id: Optional[int] = self.posdict_closed.get(cur_time, None)
            last_price = self.hist_window["Open"][-1]

            if closed_position_id is not None:
                closed_position = self.positions[closed_position_id]
                self.cumulative_profit += closed_position.profit_abs
                self.cumulative_fees += closed_position.fees_abs
                active_position = None

            if self.posdict_open.get(cur_time, None) is not None:
                active_position = self.positions[self.posdict_open[cur_time]]

            # If an active position exists, add its unrealized profit
            active_profit, active_volume, active_cost = 0, 0, 0
            if active_position is not None:
                active_profit = active_position.side.value * (last_price - active_position.open_price) * active_position.volume
                active_volume = float(active_position.volume)
                active_cost = active_position.open_price * active_volume

            max_profit = max(max_profit, self.cumulative_profit - self.cumulative_fees)

            # Append current cumulative profit to the profit curve
            self.profit_hist["dates"].append(cur_time)
            self.profit_hist["realized_pnl"].append(self.cumulative_profit)
            self.profit_hist["profit_csum_nofees"].append(self.cumulative_profit + active_profit)
            self.profit_hist["fees_csum"].append(self.cumulative_fees)
            self.profit_hist["pos_size"].append(active_volume)
            self.profit_hist["pos_cost"].append(active_cost)
            self.profit_hist["max_profit"].append(max_profit)

        self.profit_hist = pd.DataFrame(self.profit_hist)
        if not self.profit_hist.empty:
            self.profit_hist["dates"] = to_datetime(self.profit_hist["dates"])
            self.profit_hist["profit_csum"] = self.profit_hist["profit_csum_nofees"] - self.profit_hist["fees_csum"]
            self.profit_hist["realized_pnl_withfees"] = self.profit_hist["realized_pnl"] - self.profit_hist["fees_csum"]
            self.profit_hist["loss"] = self.profit_hist["realized_pnl_withfees"] - self.profit_hist["max_profit"]

    def add_info(self, wallet: float, volume_control: VolumeControl, leverage: float):
        self.wallet = wallet
        self.leverage = leverage
        self.volume_control = volume_control
        self.max_loss = max(self.profit_hist["loss"].abs())
        
        # Correct deposit for volume control rule
        if volume_control.rule == VolEstimRule.DEPOSIT_BASED:
            self.deposit = max(wallet, self.max_loss)
        elif volume_control.rule == VolEstimRule.FIXED_POS_COST:
            self.deposit = wallet + self.max_loss
        else:
            raise ValueError(f"Unknown volume control rule: {volume_control.rule}")
        if self.deposit > wallet:
            logger.warning(f"deposit was increased: {self.deposit} > {wallet}")
        logger.info(f"Export backtest results: deposit: {self.deposit}, max_loss: {self.max_loss}, leverage: {self.leverage}")

    @cached_property
    def df(self):
        return self.profit_hist
    
    @cached_property
    def ticker(self):
        tickers = set(pos.ticker for pos in self.positions)
        if len(tickers) > 1:
            raise ValueError(f"Multiple tickers in the same backtest: {tickers}")
        return tickers.pop()


class Broker:
    def __init__(self, cfg: Dict[str, Any], init_moving_window=True):
        self.symbol:Symbol = cfg["symbol"]
        self.period = cfg["period"]
        self.fee_rate = cfg["fee_rate"]
        self.leverage = cfg["leverage"]
        self.close_last_position = cfg["close_last_position"]
        self.wallet = cfg["wallet"]
        self.volume_control: VolumeControl = cfg["volume_control"]
        self.active_orders: List[Order] = []
        self.active_position: Position = None
        self.positions: List[Position] = []
        self.orders = []
        self.profit_hist: TradeHistory = None
        self.time = None
        self.hist_id = None
        self.open_price = None
        self.cumulative_profit = 0
        self.cumulative_fees = 0
        self.mw = MovingWindow(cfg) if init_moving_window else None
        self.deposit = self.wallet if self.volume_control.rule == VolEstimRule.DEPOSIT_BASED else 0
        
        self.hist_window = None

    @property
    def profits(self):
        return [p.profit for p in self.positions]

    @property
    def profits_abs(self):
        return [p.profit_abs for p in self.positions]

    @property
    def fees_abs(self):
        return [p.fees_abs for p in self.positions]

    # @profile_function
    def trade_stream(self, callback):
        for self.hist_window, _ in tqdm(self.mw(), desc="Backtest", total=self.mw.timesteps_count, disable=False):
            self.time = self.hist_window["Date"][-1]
            self.hist_id = self.hist_window["Id"][-1]
            self.open_price = self.hist_window["Open"][-1]

            closed_position = self.update()
            # Run expert and update active orders
            callback()
            closed_position_new = self.update(check_sl_tp=False)
            if closed_position is None:
                if closed_position_new is not None:
                    closed_position = closed_position_new
            elif closed_position_new is not None:
                if self.update(check_sl_tp=False) is not None:
                    raise ValueError("closed positions disagreement!")

        if self.active_position is not None and self.close_last_position:
            closed_position = self.close_active_pos(price=self.open_price,
                                      time=self.time,
                                      hist_id=self.hist_id)
            
        self.profit_hist = TradeHistory(self.mw, self.positions)
        self.profit_hist.add_info(self.wallet, self.volume_control, self.leverage)

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

    def set_active_orders(self, new_orders_list: List[Order]) -> str:
        if len(new_orders_list):
            self.close_orders(self.hist_id)
            self.active_orders = new_orders_list
        return "OK"

    @property
    def available_deposit(self):
        deposit = self.deposit
        if self.active_position is not None:
            deposit += self.active_position.unrealized_pnl(self.open_price)
        return deposit

    def update_deposit(self):
        self.deposit = min(self.wallet, self.deposit + self.active_position.profit_abs - self.active_position.fees_abs)
        assert self.volume_control.rule != VolEstimRule.DEPOSIT_BASED or self.deposit >= 0, "deposit is negative when volume control is deposit based"

    def close_active_pos(self, price, time, hist_id):
        self.active_position.close(price, time, hist_id)
        closed_position = self.active_position
        self.update_deposit()
        self.active_position = None
        self.positions.append(closed_position)
        return closed_position

    def update(self, check_sl_tp=True):
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
            triggered_id = None
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
            if (check_sl_tp and
                order.type in (ORDER_TYPE.LIMIT, ORDER_TYPE.STOPLOSS, ORDER_TYPE.TAKEPROF) and
                order.open_indx != self.hist_id):
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

            if triggered_price is not None and triggered_id is not None:
                self.close_orders(triggered_id, i)
                if self.active_position is not None:
                    if self.active_position.side.value * triggered_side.value < 0:
                        if self.active_position.set_volume(triggered_vol) >= self.active_position.volume:
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
                        ticker=self.symbol.ticker,
                        volume=triggered_vol,
                        qty_step=self.symbol.qty_step,
                        period=self.period,
                        fee_rate=self.fee_rate,
                        sl=sl,
                    )
                    triggered_vol = 0
        for order in self.active_orders:
            if order.type in (ORDER_TYPE.STOPLOSS, ORDER_TYPE.TAKEPROF):
                self.active_orders.remove(order)

        return closed_position