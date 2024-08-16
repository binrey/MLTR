import sys
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from time import perf_counter

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from data_processing.dataloading import DataParser, MovingWindow

pd.options.mode.chained_assignment = None
from multiprocessing import Process
from typing import Iterable

from backtest_broker import Broker
from experts import BacktestExpert
from real_trading import plot_fig
from utils import PyConfig

# logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

class BackTestResults:
    def __init__(self, date_start, date_end, wallet=100):
        self.date_start = date_start
        self.date_end = date_end   
        self.target_dates = [pd.to_datetime(d).date() for d in 
                             pd.date_range(start=date_start, 
                                   end=date_end, 
                                   freq="D")
                             ]
        self.daily_hist = pd.DataFrame({"days": self.target_dates})
        self.buy_and_hold = None
        self.tickers = None
        self.wallet = wallet
    
    def process_backtest(self, backtest_broker: Broker):
        t0 = perf_counter()
        self.wallet = backtest_broker.cfg.wallet if self.wallet is None else self.wallet
        profits = backtest_broker.profits_abs
        dates = [pd.to_datetime(pos.close_date).date() for pos in backtest_broker.positions]
        self.tickers = "+".join(set([pos.ticker for pos in backtest_broker.positions]))
        self.process_profits(dates, profits, backtest_broker.fees)
        self.num_years_on_trade = self.compute_n_years(backtest_broker.positions)
        # self.mean_pos_duration = np.array([pos.duration for pos in backtest_broker.positions]).mean()
        return perf_counter() - t0 
    
    def process_profits(self, dates: Iterable, profits: Iterable, fees: Iterable):
        profit_cumsum = np.array(profits).cumsum()
        self.ndeals = len(profits)
        profit_nofees = profit_cumsum + np.array(fees).cumsum()
        self.deal_hist = pd.DataFrame({"dates": dates, "profit": profit_cumsum, "profit_nofees": profit_nofees})
        # self.open_risks = np.array([pos.open_risk for pos in backtest_broker.positions])
        self.update_daily_profit(self._convert_hist(dates, profit_cumsum))
        self.fees = sum(fees)
        

    def compute_buy_and_hold(self, dates, closes, fuse=False):
        t0  = perf_counter()
        dates = [d.date() for d in pd.to_datetime(dates)]
        monthly_dates = [pd.to_datetime(d).date() for d in 
                         pd.date_range(start=self.date_start, 
                                       end=self.date_end, 
                                       freq="W")
                         ]
        bh = self._convert_hist(dates, closes, monthly_dates)
        bh = np.hstack([0, ((bh[1:] - bh[:-1])*self.wallet/bh[:-1])]).cumsum()
        self.daily_hist["buy_and_hold"] = self._convert_hist(monthly_dates, bh)
        if fuse:
            self.update_daily_profit(self.daily_hist["profit"] + self.daily_hist["buy_and_hold"])  
        return perf_counter() - t0
    
    def update_daily_profit(self, daily_profit):
        self.daily_hist["profit"] = daily_profit
        profit_stair, self.metrics = self._calc_metrics(self.daily_hist["profit"].values)
        self.deposit = self.wallet + self.metrics["loss_max"]
        self.daily_hist["deposit"] = self.deposit - (profit_stair - self.daily_hist["profit"].values)
        self.metrics["loss_max_rel"] = self.metrics["loss_max"]/self.deposit*100
        
    @property
    def final_profit(self):
        return self.daily_hist["profit"].values[-1] if len(self.daily_hist["profit"].values) > 0 else 0    
    
    @property
    def APR(self):
        final_profit_rel = self.final_profit/self.deposit*100
        return final_profit_rel/self.num_years_on_trade    
    
    @property
    def ndeals_per_year(self):
        return int(self.ndeals / self.num_years_on_trade)
    
    @property
    def ndeals_per_month(self):
        return int(self.ndeals / max(1, self.num_years_on_trade) / 12)
      
    def compute_n_years(self, positions):
        if len(positions) == 0:
            return 0
        d0 = max(np.datetime64(self.date_start), positions[0].open_date)
        d1 = min(np.datetime64(self.date_end), positions[-1].close_date)
        return (d1-d0).astype("timedelta64[M]").item()/12
    
    @staticmethod
    def _calc_metrics(ts):
        ymax = ts[0]
        twait = 0
        twaits = []
        h = np.zeros(len(ts))
        for i, y in enumerate(ts):
            if y >= ymax:
                ymax = y
                if twait>0:
                    #print(t, twait ,ymax)
                    twaits.append(twait)
                    twait = 0
            else:
                twait += 1
            h[i] = ymax
        max_loss = (h - ts).max()
        twaits.append(twait)
        twaits = np.array(twaits) if len(twaits) else np.array([len(ts)])
        twaits.sort()
        # lin_err = sum(np.abs(ts - np.arange(0, ts[-1], ts[-1]/len(ts))[:len(ts)]))
        # lin_err /= len(ts)*ts[-1]
        metrics = {"maxwait": twaits.max(),#[-5:].mean(), 
                   "recovery": ts[-1]/max_loss if max_loss else -1, 
                   "loss_max": max_loss}
        return h, metrics

    def _convert_hist(self, dates, vals, target_dates=None):
        if target_dates is None:
            target_dates = self.target_dates
        daily_vals = np.zeros(len(target_dates))
        unbias=False
        darray = np.array(dates)
        for i, date_target in enumerate(target_dates):
            # Select vals records with same day
            for _ in range(10):
                mask = darray == date_target
                if sum(mask):
                    break
                date_target -= timedelta(days=1)
            day_profs = vals[mask]
            # If there are records for currend day, store latest of them, else fill days with no records with latest sored record
            if len(day_profs):
                daily_vals[i] = day_profs[-1]
            elif len(vals):
                daily_vals[i] = daily_vals[i-1]
            else:
                pass

        if unbias:
            daily_vals = daily_vals - daily_vals[0]
        return daily_vals
    
    def metrics_from_profit(self, profit_curve):
        loss_max, wait_max = [self._calc_metrics(profit_curve)[1][k] for k in ("loss_max", "maxwait")]
        deposit = self.wallet + loss_max
        final_profit_rel = profit_curve[-1]/deposit*100
        return final_profit_rel/self.num_years_on_trade, wait_max

def find_available_date(hist: pd.DataFrame, date_start: pd.Timestamp):
    dt = hist.Date[1] - hist.Date[0]
    date_test = date_start
    while date_test not in hist.Date:
        date_test  = date_test + pd.Timedelta(days=1)
    return date_test


def backtest(cfg, loglevel = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=loglevel)
    
    exp = BacktestExpert(cfg)
    broker = Broker(cfg)
    hist_pd, hist = DataParser(cfg).load("/Users/andrybin/Yandex.Disk.localized/fin_data/")
    mw = MovingWindow(hist, cfg)

    if cfg.save_plots:
        # save_path = Path("backtests") / f"{cfg.body_classifier.func.name}-{cfg.ticker}-{cfg.period}"
        save_path = Path("backtests") / f"{cfg.ticker}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
        
    t0, texp, tbrok, tdata = perf_counter(), 0, 0, 0
    
    # for t in tqdm(range(id2start, id2end), 
    #               desc=f"back test {cfg.body_classifier.func.name}",
    #               disable=loglevel == "ERROR"):
    for (h, dt) in mw():
        t = h.Id[-1]
        tdata += dt
        texp += exp.update(h, broker.active_position)            
        
        closed_pos, dt = broker.update_state(h, exp.orders)
        exp.orders = []
        tbrok += dt
        if closed_pos is not None:
        # if broker.active_position is None and exp.order_sent:
            logger.debug(f"t = {t} -> postprocess closed position")
            if broker.active_position is None:
                broker.close_orders(h.Id[-2])
            if cfg.save_plots:
                ords_lines = [order.lines for order in broker.orders if order.open_indx >= closed_pos.open_indx]
                lines2plot = exp.lines + ords_lines + [closed_pos.lines]
                for line in lines2plot:
                    if type(line[0][0]) is pd.Timestamp:
                        continue
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if closed_pos.profit > 0 else "red"]
                widths = [1]*(len(lines2plot)-1) + [2]
                
                tplot_end = max([hist_pd.loc[t].Id if type(t) is pd.Timestamp else t for t in [line[-1][0] for line in lines2plot]])#lines2plot[-1][-1][0]
                tplot_start = min([hist_pd.loc[e[0]].Id if type(e[0]) is pd.Timestamp else e[0] for e in lines2plot[0]] + [closed_pos.lines[0][0] - cfg.hist_buffer_size])
                hist2plot = hist_pd.iloc[int(tplot_start):int(tplot_end+1)]
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
                        line[i] = (hist2plot.index[hist2plot.Id==x][0], y)
                        
                plot_fig(hist2plot=hist2plot,
                         lines2plot=lines2plot,
                         save_path=save_path,
                         prefix=cfg.ticker,
                         t=pd.to_datetime(closed_pos.open_date, utc=True),
                         side="Buy" if closed_pos.dir > 0 else "Sell",
                         ticker=cfg.ticker)
            closed_pos = None
    
    
    backtest_results = BackTestResults(mw.date_start, mw.date_end)
    tpost = backtest_results.process_backtest(broker)
    if cfg.eval_buyhold:
        tbandh = backtest_results.compute_buy_and_hold(dates=hist.Date[mw.id2start:mw.id2end],
                                                       closes=hist.Close[mw.id2start:mw.id2end], 
                                                       fuse=cfg.fuse_buyhold)
    ttotal = perf_counter() - t0
    
    sformat = lambda type: {1:"{:>30}: {:>5.0f}", 2: "{:>30}: {:5.2f}"}.get(type)
    logger.info(f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, sl={cfg.stops_processor.func.name}, sl-rate={cfg.trailing_stop_rate}")
    logger.info(sformat(2).format("total backtest", ttotal) + " sec")
    logger.info(sformat(1).format("data loadings", tdata/ttotal*100) + " %")    
    logger.info(sformat(1).format("expert updates", texp/ttotal*100) + " %")
    logger.info(sformat(1).format("broker updates", tbrok/ttotal*100) + " %")
    logger.info(sformat(1).format("postproc. broker", tpost/ttotal*100) + " %")
    if cfg.eval_buyhold:
        logger.info(sformat(1).format("Buy & Hold", tbandh/ttotal*100) + " %")

    logger.info("-"*40)
    logger.info(sformat(1).format("APR", backtest_results.APR) + f" % ({backtest_results.ndeals_per_month} deals/month)" + f" ({backtest_results.fees/backtest_results.final_profit*100:.1f}% fees)") 
    logger.info(sformat(1).format("MAXLOSS", backtest_results.metrics["loss_max_rel"]) + " %")
    logger.info(sformat(1).format("RECOVRY FACTOR", backtest_results.metrics["recovery"])) 
    logger.info(sformat(1).format("MAXWAIT", backtest_results.metrics["maxwait"]) + " days")
    # logger.info(sformat(1).format("MEAN POS. DURATION", backtest_results.mean_pos_duration) + " \n")        
    return backtest_results


if __name__ == "__main__":
    cfg = PyConfig(sys.argv[1]).test()
    btest_results = backtest(cfg, loglevel="INFO")
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()
    ax1.plot(btest_results.daily_hist.days, btest_results.daily_hist.profit, linewidth=3, color="b", alpha=0.6)
    ax1.plot(btest_results.deal_hist.dates, btest_results.deal_hist.profit, linewidth=1, color="b", alpha=0.6)    
    ax1.plot(btest_results.deal_hist.dates, btest_results.deal_hist.profit_nofees, linewidth=1, color="r", alpha=0.6)
    if "buy_and_hold" in btest_results.daily_hist.columns:
        ax1.plot(btest_results.daily_hist.days, btest_results.daily_hist.buy_and_hold, linewidth=1, alpha=0.6)  
    ax1.legend(["sum. profit", "profit from strategy", "profit without fees", "buy and hold"])
    plt.grid("on")
    ax2.plot(btest_results.daily_hist["days"], btest_results.daily_hist["deposit"], "-", linewidth=3, alpha=0.3)
    ax2.legend(["deposit"])
    
    plt.tight_layout()
    plt.savefig("backtest.png")
    plt.show()