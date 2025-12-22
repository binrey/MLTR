
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Optional

import numpy as np
import pandas as pd
import stackprinter
from loguru import logger
from tqdm import tqdm

from backtesting.backtest_broker import MultiSymbolBroker, SingleSymbolBroker
from backtesting.utils import BackTestResults
from common.visualization import get_distinct_colors
from data_processing import PULLERS
from experts.core import Expert
from trade.base import BaseTradeClass
from trade.utils import Order, Position, log_modify_sl, log_modify_tp

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


class BackTest(BaseTradeClass):
    def __init__(self, cfg) -> None:
        self.session = SingleSymbolBroker(cfg)
        super().__init__(cfg=cfg,
                         expert=Expert(cfg=cfg,
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

    def set_multisymbol_session(self, session: MultiSymbolBroker):
        self.session = session

    def get_server_time(self) -> np.datetime64:
        return self.session.time

    def get_current_position(self) -> Position:
        return self.session.active_position

    def get_all_active_positions(self) -> list[Position]:
        return self.session.active_pos_for_all_symbols

    def get_open_position(self):
        return self.session.active_position

    def get_deposit(self) -> float:
        return self.session.available_deposit

    def get_hist(self):
        return self.session.hist_window

    def get_pos_history(self):
        return self.session.positions

    def get_qty_step(self):
        return self.qty_step

    def _create_order(self, order: Order):
        self.session.set_active_order(order)
        logger.debug(f"Creating order {order.id}...OK")
        return order
        
    @log_modify_sl
    def _modify_sl(self, sl: Optional[float]):
        self.session.update_sl(sl)

    @log_modify_tp
    def _modify_tp(self, tp: Optional[float]):
        self.session.update_tp(tp)

    def postprocess(self) -> BackTestResults:
        bt_res = BackTestResults()
        bt_res.add(self.session.profit_hist)
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

    bt_res.print_results(cfg, backtest_trading.exp, use_relative=True)
    bt_res.plot_results(use_relative=True)
    bt_res.save_fig()

    return bt_res


def launch_sync_multirun(cfgs: list[dict]) -> BackTestResults:
    assert all(cfgs[0]["period"] == cfg["period"] for cfg in cfgs), "All periods must be the same"
    assert len(set((cfg["symbol"].ticker for cfg in cfgs))) == len(cfgs), "All symbols must be different. Add per expert sessins for correct work." 
    session = MultiSymbolBroker(cfgs)
    bt_res_combined = BackTestResults()
    bt_res_composition: list[BackTestResults] = []
    backtest_trading_list: list[BackTest] = []
    for cfg in cfgs:
        PULLERS["bybit"](**cfg)
        backtest_trading = BackTest(cfg)
        backtest_trading.set_multisymbol_session(session)
        backtest_trading.initialize()
        backtest_trading_list.append(backtest_trading)

    # Create per-symbol moving window generators from the underlying single-symbol brokers
    mws = [session.brokers_by_symbol[bt.ticker].mw(output_time=False) for bt in backtest_trading_list]
    # Use the shortest history among symbols to avoid StopIteration
    steps_count = min(len(session.brokers_by_symbol[bt.ticker].mw) for bt in backtest_trading_list)

    for i in tqdm(range(steps_count), desc="Backtest", total=steps_count, disable=False):
        closed_positions = []
        for bt, mw in zip(backtest_trading_list, mws):
            session.switch_symbol(bt.ticker)
            closed_positions.append(session.pre_expert_update(next(mw)))
        for bt in backtest_trading_list:
            session.switch_symbol(bt.ticker)
            bt.handle_trade_message()
        for bt, closed_position in zip(backtest_trading_list, closed_positions):
            session.switch_symbol(bt.ticker)
            session.post_expert_update(closed_position)

    for bt in backtest_trading_list:
        session.switch_symbol(bt.ticker)
        session.stream_postprocess()
        bt_res = BackTestResults()
        bt_res.add(deepcopy(session.profit_hist))
        bt_res.eval_daily_metrics()
        bt_res.print_results()
        bt_res_composition.append(bt_res)
        bt_res_combined.add(session.profit_hist)
        
    bt_res_combined.eval_daily_metrics()
    bt_res_combined.print_results()
    bt_res_combined.plot_results(plot_profit_without_fees=False)
    
    colors = get_distinct_colors(len(bt_res_composition))
    for i, bt_res in enumerate(bt_res_composition):
        bt_res_combined.add_from_other_results(btest_results=bt_res, color=colors[i % len(colors)])
    bt_res_combined.save_fig()
    return bt_res_combined


def launch_multirun(cfgs: list[dict]) -> BackTestResults:
    bt_res_combined = BackTestResults()
    bt_res_composition = []

    for cfg in cfgs:
        backtest_trading = BackTest(cfg)
        backtest_trading.initialize()
        backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
        
        bt_res = BackTestResults()
        bt_res.add(deepcopy(backtest_trading.session.profit_hist))
        bt_res.eval_daily_metrics()
        bt_res.print_results()
        bt_res_composition.append(bt_res)
        bt_res_combined.add(backtest_trading.session.profit_hist)

    if len(bt_res_composition) > 1:
        bt_res_combined.eval_daily_metrics()
        bt_res_combined.print_results()
        bt_res_combined.plot_results(plot_profit_without_fees=False)
        
        colors = get_distinct_colors(len(bt_res_composition))
        for i, bt_res in enumerate(bt_res_composition):
            bt_res_combined.add_from_other_results(btest_results=bt_res, color=colors[i % len(colors)])
        bt_res_combined.save_fig()
        return bt_res_combined



