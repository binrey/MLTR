import sys
from pathlib import Path
from shutil import rmtree
from time import perf_counter
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
# import yfinance as yf
from loguru import logger
from tqdm import tqdm
from dataloading import MovingWindow, DataParser
pd.options.mode.chained_assignment = None
from utils import PyConfig
from backtest_broker import Broker
from experts import BacktestExpert
from real_trading import plot_fig
from multiprocessing import Process
# logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

class BackTestResults:
    def __init__(self, date_start, date_end):
        self.date_start = date_start
        self.date_end = date_end   
        dates2load = pd.date_range(start="/".join([date_start.split("-")[i] for i in [1, 2, 0]]), 
                                   end="/".join([date_end.split("-")[i] for i in [1, 2, 0]]), 
                                   freq="D")
        self.target_dates = [pd.to_datetime(d).date() for d in dates2load]
        self.daily_hist = pd.DataFrame({"days": self.target_dates})
        self.buy_and_hold = None
        self.cfg = None
        
    def process_backtest(self, backtest_broker: Broker):
        t0 = perf_counter()
        self.cfg = backtest_broker.cfg
        self.profits = backtest_broker.profits_abs
        profit_cumsum = self.profits.cumsum()
        dates = [pd.to_datetime(pos.close_date).date() for pos in backtest_broker.positions]
        profit_nofees = profit_cumsum + backtest_broker.fees.cumsum()
        self.deal_hist = pd.DataFrame({"dates": dates, "profit": profit_cumsum, "profit_nofees": profit_nofees})
        self.ndeals = len(self.profits)
        self.mean_pos_duration = np.array([pos.duration for pos in backtest_broker.positions]).mean()
        # self.open_risks = np.array([pos.open_risk for pos in backtest_broker.positions])
        self.num_years_on_trade = self.compute_n_years(backtest_broker)
        self.update_daily_profit(self._convert_hist(profit_cumsum, dates))
        self.fees = sum(backtest_broker.fees)
        return perf_counter() - t0

    def compute_buy_and_hold(self, closes, dates, fuse=False):
        t0  = perf_counter()
        dates = [d.date() for d in pd.to_datetime(dates)]
        bh = self._convert_hist(closes, dates)
        bh = np.hstack([0, ((bh[1:] - bh[:-1])*self.cfg.wallet/bh[:-1])]).cumsum()
        self.daily_hist["buy_and_hold"] = bh
        if fuse:
            self.update_daily_profit(self.daily_hist["profit"] + bh)  
        return perf_counter() - t0
    
    def update_daily_profit(self, daily_profit):
        self.daily_hist["profit"] = daily_profit
        profit_stair = self._calc_metrics()
        self.deposit = self.cfg.wallet + self.metrics["loss_max"]
        self.daily_hist["deposit"] = self.deposit - (profit_stair - self.daily_hist["profit"].values)
        self.metrics["loss_max_rel"] = self.metrics["loss_max"]/self.deposit*100
        
    @property
    def final_profit(self):
        return self.daily_hist["profit"].values[-1] if len(self.daily_hist["profit"].values) > 0 else 0    
    
    @property
    def APR(self):
        final_profit_rel = self.final_profit/self.deposit*100
        return final_profit_rel/self.num_years_on_trade    
      
    def compute_n_years(self, backtest_broker):
        d0 = backtest_broker.positions[0].open_date
        d1 = backtest_broker.positions[-1].close_date
        return (d1-d0).astype("timedelta64[M]").item()/12
    
    def _calc_metrics(self):
        ts = self.daily_hist["profit"].values
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
        self.metrics = {"maxwait": twaits.max(),#[-5:].mean(), 
                   "recovery": ts[-1]/max_loss, 
                   "loss_max": max_loss}
        return h

    def _convert_hist(self, vals, dates):
        daily_vals = np.zeros(len(self.target_dates))
        unbias=False
        darray = np.array(dates)
        for i, date_terget in enumerate(self.target_dates):
            # Select vals records with same day
            day_profs = vals[darray == date_terget]
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
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg.hist_buffer_size)

    if cfg.save_plots:
        save_path = Path("backtests") / f"{cfg.body_classifier.func.name}-{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    date_start = np.datetime64(cfg.date_start)
    mask = hist.Date == date_start
    id2start = 0
    if sum(mask) == 1:
        id2start = hist.Id[mask][0]
    else:
        logger.warning(f"!!! Date start {date_start} not found, current dates range {hist.Date[0]} - {hist.Date[-1]}")
        
    if id2start < cfg.hist_buffer_size:
        logger.warning(f"Not enough history, shift start id from {id2start} to {cfg.hist_buffer_size}")
        id2start = cfg.hist_buffer_size
        logger.warning(f"!!! Switch to {hist.Date[id2start]}")
        
    id2end = cfg.tend if cfg.tend is not None else hist.Id.shape[0]
    t0, texp, tbrok, tdata = perf_counter(), 0, 0, 0
    
    for t in tqdm(range(id2start, id2end), 
                  desc=f"back test {cfg.body_classifier.func.name}",
                  disable=loglevel == "ERROR"):
        h, dt = mw(t)
        tdata += dt
        texp += exp.update(h, broker.active_position)
        if len(exp.orders):
            broker.active_orders = exp.orders
            exp.orders = []
        
        closed_pos, dt = broker.update(h)
        tbrok += dt
        # if closed_pos is not None:
        if broker.active_position is None and exp.order_sent:
            logger.debug(f"t = {t} -> postprocess closed position")
            broker.close_orders(h.Id[-2])
            if cfg.save_plots:
                ords_lines = [order.lines for order in broker.orders if order.open_indx >= closed_pos.open_indx]
                lines2plot = exp.lines + ords_lines + [closed_pos.lines]
                for line in lines2plot:
                    if len(line) > 2:
                        while line[0][0] < closed_pos.lines[0][0] - cfg.hist_buffer_size:
                            line.pop(0)
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if closed_pos.profit > 0 else "red"]
                widths = [1]*(len(lines2plot)-1) + [2]
                
                tplot_end = lines2plot[-1][-1][0]
                tplot_start = min([e[0] for e in lines2plot[0]] + [closed_pos.lines[0][0] - cfg.hist_buffer_size])
                hist2plot = hist_pd.iloc[tplot_start:tplot_end+1]
                min_id = hist2plot.Id.min()
                for line in lines2plot:
                    for i, point in enumerate(line):
                        x, y = point
                        x = max(x, min_id)
                        try:
                            y = y.item()
                        except:
                            pass
                        line[i] = (hist2plot.index[hist2plot.Id==x][0], y)
                        
                # p = Process(target=plot_fig, args=(hist2plot, lines2plot, save_path, cfg.ticker, 
                #                                    pd.to_datetime(closed_pos.open_date, utc=True),
                #                                    "Buy" if closed_pos.dir > 0 else "Sell",
                #                                     cfg.ticker))
                # p.start()
                # p.join()
                plot_fig(hist2plot=hist2plot,
                         lines2plot=lines2plot,
                         save_path=save_path,
                         prefix=cfg.ticker,
                         t=pd.to_datetime(closed_pos.open_date, utc=True),
                         side="Buy" if closed_pos.dir > 0 else "Sell",
                         ticker=cfg.ticker)

    
    
    backtest_results = BackTestResults(cfg.date_start, cfg.date_end)
    tpost = backtest_results.process_backtest(broker)
    if cfg.eval_buyhold:
        tbandh = backtest_results.compute_buy_and_hold(hist.Close[id2start: id2end], 
                                                       hist.Date[id2start: id2end],
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
        logger.info(sformat(1).format("But & Hold", tbandh/ttotal*100) + " %")

    logger.info("-"*30)
    logger.info(sformat(1).format("APR", backtest_results.APR) + f" % ({backtest_results.ndeals} deals)" + f" ({backtest_results.fees/backtest_results.final_profit*100:.1f}% fees)") 
    logger.info(sformat(1).format("MAXLOSS", backtest_results.metrics["loss_max_rel"]) + "%")
    logger.info(sformat(1).format("RECOVRY FACTOR", backtest_results.metrics["recovery"])) 
    logger.info(sformat(1).format("MAXWAIT", backtest_results.metrics["maxwait"]) + " days")
    logger.info(sformat(1).format("MEAN POS. DURATION", backtest_results.mean_pos_duration) + " \n")        
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
    plt.legend(["profit", "profit without fees", "buy and hold"])
    ax2.plot(btest_results.daily_hist["days"], btest_results.daily_hist["deposit"], "-", linewidth=3, alpha=0.3)
    plt.legend(["deposit"])
    
    plt.tight_layout()
    plt.grid("on")
    plt.savefig("backtest.png")
    plt.show()