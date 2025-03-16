from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import stackprinter
from loguru import logger

from backtesting.backtest_broker import Broker
from backtesting.utils import BackTestResults
from experts.experts import BacktestExpert
from trade.base import BaseTradeClass, log_get_hist
from trade.utils import Position

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


class BackTest(BaseTradeClass):
    def __init__(self, cfg, backtest_session: Broker) -> None:
        self.session = backtest_session
        super().__init__(cfg=cfg, expert=BacktestExpert(cfg=cfg, session=backtest_session))

    def get_server_time(self) -> np.datetime64:
        return self.session.time

    def get_current_position(self) -> Position:
        return self.session.active_position

    @log_get_hist
    def get_hist(self):
        return self.session.hist_window

    def get_pos_history(self):
        return self.session.positions

    def get_qty_step(self):
        return self.cfg["equaty_step"]

    def postprocess(self) -> BackTestResults:
        bt_res = BackTestResults(self.session.mw.date_start, self.session.mw.date_end)
        bt_res.process_backtest(self.session)
        if bt_res.ndeals == 0:
            logger.warning("No trades!")
            return bt_res

        if self.cfg['eval_buyhold']:
            bt_res.compute_buy_and_hold(
                dates=self.session.mw.hist["Date"][self.session.mw.id2start: self.session.mw.id2end],
                closes=self.session.mw.hist["Close"][self.session.mw.id2start: self.session.mw.id2end],
            )
        return bt_res

def launch(cfg) -> BackTestResults:
    if cfg["save_plots"]:
        save_path = Path("backtests") / f"{cfg['symbol'].ticker}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir(parents=True)

    bt_session = Broker(cfg)
    backtest_trading = BackTest(cfg=cfg, backtest_session=bt_session)
    backtest_trading.test_connection()
    print()
    backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
    bt_res = backtest_trading.postprocess()

    bt_res.print_results(cfg, backtest_trading.exp)
    if cfg["save_plots"]:
        bt_res.plot_results()
    
    return bt_res


def launch_multirun(cfgs: list[dict]):
    # run backtest for each config and create combined profit_hist
    daily_hist = None
    for cfg in cfgs:
        bt_res = launch(cfg)
        if daily_hist is None:
            daily_hist = bt_res.daily_hist
        else:
            daily_hist += bt_res.daily_hist
        
    # create new backtest result with all positions from all results
    bt_res_combined = BackTestResults(cfgs[0]["date_start"], cfgs[0]["date_end"])
    bt_res_combined.wallet = bt_res.wallet
    bt_res_combined.process_profit_hist(daily_hist)
    bt_res_combined.plot_results()
    return bt_res_combined
