import os
import sys
from pathlib import Path
from shutil import rmtree
from time import perf_counter

import pandas as pd
from loguru import logger

from data_processing.dataloading import DataParser, MovingWindow

from .utils import BackTestResults

pd.options.mode.chained_assignment = None

from backtesting.backtest_broker import Broker
from experts import BacktestExpert


def find_available_date(hist: pd.DataFrame, date_start: pd.Timestamp):
    dt = hist.Date[1] - hist.Date[0]
    date_test = date_start
    while date_test not in hist.Date:
        date_test = date_test + pd.Timedelta(days=1)
    return date_test


def backtest(cfg, loglevel="INFO"):
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    exp = BacktestExpert(cfg)
    broker = Broker(cfg)
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg)

    if cfg.save_plots:
        # save_path = Path("backtests") / f"{cfg.body_classifier.func.name}-{cfg.ticker}-{cfg.period}"
        save_path = Path("backtests") / f"{cfg.ticker}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()

    t0, texp, tbrok, tdata = perf_counter(), 0, 0, 0

    for h, dt in mw():
        t = h.Id[-1]
        tdata += dt
        texp += exp.update(h, broker.active_position)

        closed_pos, dt = broker.update_state(h, exp.orders)
        exp.orders = []
        tbrok += dt
        if closed_pos is not None:
            # if broker.active_position is None and exp.order_sent:
            logger.info(f"t = {t} -> postprocess closed position")
            if broker.active_position is None:
                broker.close_orders(h.Id[-2])
            if cfg.save_plots:
                ords_lines = [
                    order.lines
                    for order in broker.orders
                    if order.open_indx >= closed_pos.open_indx
                ]
                lines2plot = exp.lines + ords_lines + [closed_pos.lines]
                for line in lines2plot:
                    if type(line[0][0]) is pd.Timestamp:
                        continue
                colors = ["blue"] * (len(lines2plot) - 1) + [
                    "green" if closed_pos.profit > 0 else "red"
                ]
                widths = [1] * (len(lines2plot) - 1) + [2]

                tplot_end = max(
                    [
                        hist_pd.loc[t].Id if type(t) is pd.Timestamp else t
                        for t in [line[-1][0] for line in lines2plot]
                    ]
                )  # lines2plot[-1][-1][0]
                tplot_start = min(
                    [
                        hist_pd.loc[e[0]].Id if type(e[0]) is pd.Timestamp else e[0]
                        for e in lines2plot[0]
                    ]
                    + [closed_pos.lines[0][0] - cfg.hist_buffer_size]
                )
                hist2plot = hist_pd.iloc[int(tplot_start) : int(tplot_end + 1)]
                min_id = hist2plot.Id.min()
                for line in lines2plot:
                    for i, point in enumerate(line):
                        x, y = point
                        if type(x) is pd.Timestamp:
                            continue
                        x = max(x, min_id)
                        try:
                            y = y.item()
                        except:
                            pass
                        line[i] = (hist2plot.index[hist2plot.Id == x][0], y)

                plot_fig(
                    hist2plot=hist2plot,
                    lines2plot=lines2plot,
                    save_path=save_path,
                    prefix=cfg.ticker,
                    time=pd.to_datetime(closed_pos.open_date, utc=True),
                    side=closed_pos.side,
                    ticker=cfg.ticker,
                )
            closed_pos = None

    bt_res = BackTestResults(mw.date_start, mw.date_end)
    tpost = bt_res.process_backtest(broker)
    if cfg.eval_buyhold:
        tbandh = bt_res.compute_buy_and_hold(
            dates=hist.Date[mw.id2start : mw.id2end],
            closes=hist.Close[mw.id2start : mw.id2end],
            fuse=cfg.fuse_buyhold,
        )
    ttotal = perf_counter() - t0

    sformat = lambda nd: "{:>30}: {:>5.@f}".replace("@", str(nd))

    logger.info(
        f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, "
        f"sl={cfg.stops_processor.func.name}, sl-rate={cfg.trailing_stop_rate}"
    )

    logger.info(sformat(1).format("total backtest", ttotal) + " sec")
    logger.info(sformat(1).format("data loadings", tdata / ttotal * 100) + " %")
    logger.info(sformat(1).format("expert updates", texp / ttotal * 100) + " %")
    logger.info(sformat(1).format("broker updates", tbrok / ttotal * 100) + " %")
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
    return bt_res