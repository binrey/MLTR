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
logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

class BackTestResults:
    def __init__(self, backtest_broker, date_start, date_end):
        self.cfg = backtest_broker.cfg
        self.profits = backtest_broker.profits
        self.balance = self.profits.cumsum()
        self.ndeals = len(self.profits)
        self.durations = np.array([pos.duration for pos in backtest_broker.positions])
        self.open_risks = np.array([pos.open_risk for pos in backtest_broker.positions])
        self.dates = [pd.to_datetime(pos.close_date).date() for pos in backtest_broker.positions]
        self.final_balance = self.balance[-1]
        self.daily_balance = self.convert_hist(self.profits, self.dates, date_start, date_end)
        self.daily_bstair, self.metrics = self._calc_metrics(self.daily_balance["balance"])
        self.metrics.update({"mean_pos_duration": self.durations.mean(),
                             "mean_pos_result": self.profits.mean(),
                             "mean_open_risk": np.nanmean(self.open_risks)
                             })

    @staticmethod
    def _calc_metrics(ts):
        ymax = ts[0]
        twait = 0
        twaits = []
        h = []
        for y in ts:
            if y >= ymax:
                ymax = y
                if twait>0:
                    #print(t, twait ,ymax)
                    twaits.append(twait)
                    twait = 0
            else:
                twait += 1
            h.append(ymax)
        h = np.array(h)
        max_loss = (h - ts).max()
        twaits.append(twait)
        twaits = np.array(twaits) if len(twaits) else np.array([len(ts)])
        twaits.sort()
        lin_err = sum(np.abs(ts - np.arange(0, ts[-1], ts[-1]/len(ts))[:len(ts)]))
        lin_err /= len(ts)*ts[-1]
        metrics = {"maxwait": twaits.max(),#[-5:].mean(), 
                   "recovery": ts[-1]/max_loss, 
                   "loss_max": max_loss}
        return h, metrics

    @staticmethod
    def convert_hist(profits, dates, t0, t1):
        dates2load = pd.date_range(start="/".join([t0.split("-")[i] for i in [1, 2, 0]]), 
                                   end="/".join([t1.split("-")[i] for i in [1, 2, 0]]), 
                                   freq="D")
        target_dates = [pd.to_datetime(d).date() for d in dates2load]
        profit = profits.cumsum()
        balance = [0]
        unbias=False
        for d in target_dates:
            # Select balance records with same day
            day_profs = [balance[-1]] + [b for b, sd in zip(profit, dates) if sd == d]
            # If there are records for currend day, store latest of them, else fill days with no records with latest sored record
            balance.append(day_profs[-1])
        if len(balance) - len(target_dates) == 1:
            target_dates = [target_dates[0]] + target_dates
        if len(target_dates) + len(balance) == 1:
            balance = balance + [balance[-1]] 
        res = np.array(balance)
        if unbias:
            res = res - profit[0]
        return {"days": target_dates, "balance": res}


def backtest(cfg):
    exp = BacktestExpert(cfg)
    broker = Broker(cfg)
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg.hist_buffer_size)
    if cfg.save_plots:
        save_path = Path("backtests") / f"{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    tstart = max(cfg.hist_buffer_size+1, cfg.tstart)
    tend = cfg.tend if cfg.tend is not None else hist.Id.shape[0]
    t0, texp, tbrok, tdata = perf_counter(), 0, 0, 0
    for t in tqdm(range(tstart, tend), "back test"):
        h, dt = mw(t)
        # TODO
        # if h.Date[-1].astype(np.datetime64) < np.array("2024-02-15T00:30:00", dtype=np.datetime64):
        #     continue
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
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if closed_pos.profit > 0 else "red"]
                widths = [1]*(len(lines2plot)-1) + [2]
                
                t1plot = lines2plot[-1][-1][0]
                t0plot = min([e[0] for e in lines2plot[0]]) 
                dbars = max(0, 64 - (t1plot - t0plot))
                hist2plot = hist_pd.iloc[t0plot - dbars:lines2plot[-1][-1][0]+1]
                for line in lines2plot:
                    for i, point in enumerate(line):
                        y = point[1]
                        try:
                            y = y.item()
                        except:
                            pass
                        line[i] = (hist2plot.index[hist2plot.Id==point[0]][0], y)
                        
                p = Process(target=plot_fig, args=(hist2plot, lines2plot, save_path, cfg.ticker, 
                                                   pd.to_datetime(closed_pos.open_date, utc=True),
                                                   "Buy" if closed_pos.dir > 0 else "Sell",
                                                    cfg.ticker))
                p.start()
                p.join()
                # plot_fig(hist2plot=hist2plot,
                #          lines2plot=lines2plot,
                #          save_path=save_path,
                #          prefix=cfg.ticker,
                #          t=pd.to_datetime(closed_pos.open_date, utc=True),
                #          side="Buy" if closed_pos.dir > 0 else "Sell",
                #          ticker=cfg.ticker)

    
    ttotal = perf_counter() - t0
    backtest_results = BackTestResults(broker, cfg.date_start, cfg.date_end)
    sformat = lambda type: {1:"{:>30}: {:>5.0f}", 2: "{:>30}: {:5.2f}"}.get(type)
    logger.info(f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, sl={cfg.stops_processor.func.name}, sl-rate={cfg.trailing_stop_rate}")
    logger.info(sformat(2).format("total backtest", ttotal) + " sec")
    logger.info(sformat(1).format("expert updates", texp/ttotal*100) + " %")
    logger.info(sformat(1).format("broker updates", tbrok/ttotal*100) + " %")
    logger.info(sformat(1).format("data loadings", tdata/ttotal*100) + " %")
    logger.info("-"*30)
    logger.info(sformat(1).format("FINAL PROFIT", backtest_results.final_balance) + f" %  ({backtest_results.ndeals} deals)") 
    logger.info(sformat(2).format("MEAN POS. RESULT", backtest_results.metrics["mean_pos_result"]) + " %")
    logger.info(sformat(2).format("MEAN ONOPEN RISK", backtest_results.metrics["mean_open_risk"]) + " %")
    logger.info(sformat(1).format("MEAN POS. DURATION", backtest_results.metrics["mean_pos_duration"]))            
    logger.info(sformat(1).format("RECOVRY FACTOR", backtest_results.metrics["recovery"])) 
    logger.info(sformat(1).format("MAXWAIT", backtest_results.metrics["maxwait"])+"\n")
    
    # import pickle
    # pickle.dump((cfg, broker), open(str(Path("backtests") / f"btest{0:003.0f}.pickle"), "wb"))
    return backtest_results
    
    
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    cfg = PyConfig(sys.argv[1]).test()
    btest_results = backtest(cfg)
    plt.subplots(figsize=(15, 8))
    plt.plot(btest_results.dates, btest_results.balance, linewidth=2, alpha=0.6)
    plt.tight_layout()
    plt.savefig("backtest.png")