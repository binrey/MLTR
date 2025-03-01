from trade.base import BaseTradeClass, log_get_hist
from experts.experts import BacktestExpert
from backtesting.backtest_broker import Broker
import stackprinter
import numpy as np
from pathlib import Path
from shutil import rmtree

import pandas as pd
from loguru import logger

from backtesting.utils import BackTestResults
from trade.utils import Position

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


class BackTest(BaseTradeClass):
    def __init__(self, cfg, expert, telebot, session: Broker) -> None:
        super().__init__(cfg=cfg, expert=expert, telebot=telebot)
        self.session = session

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


def launch(cfg):
    if cfg["save_plots"]:
        save_path = Path("backtests") / f"{cfg['symbol'].ticker}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir(parents=True)

    bt_session = Broker(cfg)
    backtest_trading = BackTest(
        cfg=cfg,
        expert=BacktestExpert(cfg=cfg, session=bt_session),
        telebot=None,
        session=bt_session
    )
    backtest_trading.test_connection()
    print()
    bt_session.trade_stream(backtest_trading.handle_trade_message)

    bt_res = BackTestResults(bt_session.mw.date_start, bt_session.mw.date_end)
    bt_res.process_backtest(bt_session)
    if bt_res.ndeals == 0:
        logger.warning("No trades!")
        return bt_res

    if cfg['eval_buyhold']:
        bt_res.compute_buy_and_hold(
            dates=bt_session.mw.hist["Date"][bt_session.mw.id2start: bt_session.mw.id2end],
            closes=bt_session.mw.hist["Close"][bt_session.mw.id2start: bt_session.mw.id2end],
        )

    def sformat(nd): return "{:>30}: {:>5.@f}".replace("@", str(nd))

    logger.info(
        f"{cfg['symbol'].ticker}-{cfg['period']}: {backtest_trading.exp}"
    )

    logger.info("-" * 40)
    logger.info(sformat(0).format("APR", bt_res.APR) + f" %")
    logger.info(
        sformat(0).format("FINAL PROFIT", bt_res.final_profit_rel)
        + f" %"
        + f" ({bt_res.fees/bt_res.final_profit*100:.1f}% fees)"
    )
    logger.info(
        sformat(2).format("DEALS/MONTH", bt_res.ndeals_per_month)
        + f"   ({bt_res.ndeals} total)"
    )
    logger.info(sformat(0).format(
        "MAXLOSS", bt_res.metrics["loss_max_rel"]) + " %")
    logger.info(sformat(0).format(
        "RECOVRY FACTOR", bt_res.metrics["recovery"]))
    logger.info(sformat(0).format(
        "MAXWAIT", bt_res.metrics["maxwait"]) + " days")

    bt_res.plot_results()
    return bt_res
