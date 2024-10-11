from pathlib import Path
from shutil import rmtree
from time import time

import pandas as pd
from loguru import logger

from backtesting.utils import BackTestResults
from data_processing.dataloading import MovingWindow
from trade.utils import Position

pd.options.mode.chained_assignment = None

import numpy as np
import pandas as pd
import stackprinter
import yaml

from backtesting.backtest_broker import Broker
from experts import BacktestExpert
from trade.base import BaseTradeClass, log_get_hist

stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


class BackTest(BaseTradeClass):
    def __init__(self, cfg, expert, telebot, session: Broker) -> None:
        super().__init__(cfg=cfg, expert=expert, telebot=telebot)
        self.session = session
            
    def get_server_time(self) -> np.datetime64:
        return self.session.time
        
    def update_trailing_stop(self, sl_new: float) -> None:
        pass

    def get_current_position(self) -> Position:
        return self.session.active_position
    
    @log_get_hist
    def get_hist(self):
        return self.session.hist_window
    
    def get_pos_history(self):
        return self.session.positions

def launch(cfg):
    t0 = time()
    with open("./api.yaml", "r") as f:
        creds = yaml.safe_load(f)

    if cfg.save_plots:
        # save_path = Path("backtests") / f"{cfg.body_classifier.func.name}-{cfg.ticker}-{cfg.period}"
        save_path = Path("backtests") / f"{cfg.ticker}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir(parents=True)
    mw = MovingWindow(cfg)
    backtest_session = Broker(cfg)
    backtest_trading = BackTest(
        cfg=cfg, 
        expert=BacktestExpert(cfg=cfg, session=backtest_session), 
        telebot=None,
        session=backtest_session
        )
    backtest_trading.test_connection()
    
    print()
    backtest_session.trade_stream(backtest_trading.handle_trade_message)

    bt_res = BackTestResults(mw.date_start, mw.date_end)
    tpost = bt_res.process_backtest(backtest_session)
    if cfg.eval_buyhold:
        tbandh = bt_res.compute_buy_and_hold(
            dates=mw.hist["Date"][mw.id2start : mw.id2end],
            closes=mw.hist["Close"][mw.id2start : mw.id2end],
            fuse=cfg.fuse_buyhold,
        )
    ttotal = time() - t0

    sformat = lambda nd: "{:>30}: {:>5.@f}".replace("@", str(nd))

    logger.info(
        f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, "
        f"sl={cfg.sl_processor.func.name}, sl-rate={cfg.trailing_stop_rate}"
    )

    logger.info(sformat(1).format("total backtest", ttotal) + " sec")
    # logger.info(sformat(1).format("data loadings", tdata / ttotal * 100) + " %")
    # logger.info(sformat(1).format("expert updates", texp / ttotal * 100) + " %")
    # logger.info(sformat(1).format("broker updates", tbrok / ttotal * 100) + " %")
    logger.info(sformat(1).format("postproc. broker", tpost / ttotal * 100) + " %")

    if cfg.eval_buyhold:
        logger.info(sformat(1).format("Buy & Hold", tbandh / ttotal * 100) + " %")

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
    logger.info(sformat(0).format("MAXLOSS", bt_res.metrics["loss_max_rel"]) + " %")
    logger.info(sformat(0).format("RECOVRY FACTOR", bt_res.metrics["recovery"]))
    logger.info(sformat(0).format("MAXWAIT", bt_res.metrics["maxwait"]) + " days")
    # logger.info(sformat(1).format("MEAN POS. DURATION", bt_res.mean_pos_duration) + " \n")
    
    bt_res.plot_results()
    
    