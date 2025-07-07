import random
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Optional

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import stackprinter
from loguru import logger

from backtesting.backtest_broker import Broker
from backtesting.utils import BackTestResults
from data_processing import PULLERS
from experts.core import Expert
from trade.base import BaseTradeClass, log_get_hist
from trade.utils import (
    ORDER_TYPE,
    Order,
    Position,
    log_creating_order,
    log_modify_sl,
    log_modify_tp,
)

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


class BackTest(BaseTradeClass):
    def __init__(self, cfg) -> None:
        self.session = Broker(cfg)
        super().__init__(cfg=cfg,
                         expert=Expert(cfg=cfg,
                                      create_orders_func=self.create_orders,
                                      modify_sl_func=self.modify_sl,
                                      modify_tp_func=self.modify_tp),
                         telebot=None)
        self.init_save_path()

    def init_save_path(self):
        if self.save_plots:
            save_path = Path("backtests") / f"{self.ticker}"
            if save_path.exists():
                rmtree(save_path)
            save_path.mkdir(parents=True)

    def get_server_time(self) -> np.datetime64:
        return self.session.time

    def get_current_position(self) -> Position:
        return self.session.active_position

    def get_open_position(self):
        return self.session.active_position

    def get_deposit(self) -> float:
        return self.session.available_deposit

    @log_get_hist
    def get_hist(self):
        return self.session.hist_window

    def get_pos_history(self):
        return self.session.positions

    def get_qty_step(self):
        return self.qty_step

    @log_creating_order
    def _create_orders(self, side, volume, time_id):
        orders = [Order(0, side, ORDER_TYPE.MARKET, volume, time_id, time_id)]
        self.session.set_active_orders(orders)
        return orders
        
    @log_modify_sl
    def _modify_sl(self, sl: Optional[float]):
        self.session.update_sl(sl)

    @log_modify_tp
    def _modify_tp(self, tp: Optional[float]):
        self.session.update_tp(tp)

    def postprocess(self) -> BackTestResults:
        bt_res = BackTestResults()
        bt_res.add(self.session)
        bt_res.eval_daily_metrics()
        if bt_res.ndeals == 0:
            logger.warning("No trades!")
            return bt_res
        return bt_res

def launch(cfg) -> BackTestResults:
    PULLERS["bybit"](**cfg)
    backtest_trading = BackTest(cfg)
    backtest_trading.initialize()
    
    backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
    bt_res = backtest_trading.postprocess()

    bt_res.print_results(cfg, backtest_trading.exp, use_relative=False)
    bt_res.plot_results(use_relative=False)
    bt_res.save_fig()

    return bt_res


def launch_multirun(cfgs: list[dict]) -> BackTestResults:
    bt_res_combined = BackTestResults()
    bt_res_composition = []

    for cfg in cfgs:
        backtest_trading = BackTest(cfg)
        backtest_trading.initialize()
        backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
        
        bt_res = BackTestResults()
        bt_res.add(deepcopy(backtest_trading.session))
        bt_res.eval_daily_metrics()
        bt_res.print_results()
        bt_res_composition.append(bt_res)
        bt_res_combined.add(backtest_trading.session)
        
    bt_res_combined.eval_daily_metrics()
    bt_res_combined.print_results()
    bt_res_combined.plot_results(plot_profit_without_fees=False)
    
    colors = get_distinct_colors(len(bt_res_composition))
    for i, bt_res in enumerate(bt_res_composition):
        bt_res_combined.add_from_other_results(btest_results=bt_res, color=colors[i % len(colors)])
    bt_res_combined.save_fig()
    return bt_res_combined


def get_distinct_colors(num_colors):
    """
    Generate a list of distinct colors for plotting multiple curves.
    
    Args:
        num_colors: Number of distinct colors needed
        
    Returns:
        List of color values
    """
    colors = list(mcolors.TABLEAU_COLORS.values())
    # If we need more colors than available in TABLEAU_COLORS, add more from CSS4_COLORS
    if num_colors > len(colors):
        more_colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(more_colors)
        colors.extend(more_colors)
    
    return colors
