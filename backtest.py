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
from experts import BacktestExpert, PyConfig
from utils import Broker
logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

class BackTestResults:
    def __init__(self, backtest_broker, date_start, date_end):
        self.cfg = backtest_broker.cfg
        self.profits = backtest_broker.profits
        self.balance = self.profits.cumsum()
        self.ndeals = len(self.profits)
        self.dates = [pd.to_datetime(pos.close_date).date() for pos in backtest_broker.positions]
        self.final_balance = self.balance[-1]
        self.daily_balance = self.convert_hist(self.profits, self.dates, date_start, date_end)
        self.daily_bstair, self.metrics = self.calc_metrics(self.daily_balance)

    @staticmethod
    def calc_metrics(ts):
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
                   "linearity": ts[-1]/max_loss, 
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
        res = np.array(balance)
        if unbias:
            res = res - profit[0]
        return res


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
        tdata += dt
        if t < tstart or len(broker.active_orders) == 0:
            texp += exp.update(h)
            broker.active_orders = exp.orders
        
        pos, dt = broker.update(h)
        tbrok += dt
        if pos is not None:
            logger.debug(f"t = {t} -> postprocess closed position")
            broker.close_orders(h.Id[-2])
            if cfg.save_plots:
                ords_lines = [order.lines for order in broker.orders if order.open_indx >= pos.open_indx]
                lines2plot = exp.lines + ords_lines + [pos.lines]
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if pos.profit > 0 else "red"]
                widths = [1]*(len(lines2plot)-1) + [2]
                
                hist2plot = hist_pd.iloc[lines2plot[0][0][0]:lines2plot[-1][-1][0]+1]
                for line in lines2plot:
                    for i, point in enumerate(line):
                        y = point[1]
                        try:
                            y = y.item()
                        except:
                            pass
                        line[i] = (hist2plot.index[hist2plot.Id==point[0]][0], y)
                
                fig = mpf.plot(hist2plot, 
                            type='candle', 
                            block=False,
                            alines=dict(alines=lines2plot, colors=colors, linewidths=widths),
                            savefig=save_path / f"fig-{str(pos.open_date).split('.')[0]}.png")
                del fig
            exp.reset_state()
    
    ttotal = perf_counter() - t0
    backtest_results = BackTestResults(broker, cfg.date_start, cfg.date_end)
    sformat = "{:>30}: {:>3.0f} %"
    logger.info(f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, sl={cfg.stops_processor.func.name}, sl-rate={cfg.trailing_stop_rate}")
    logger.info("{:>30}: {:.1f} sec".format("total backtest", ttotal))
    logger.info(sformat.format("expert updates", texp/ttotal*100))
    logger.info(sformat.format("broker updates", tbrok/ttotal*100))
    logger.info(sformat.format("data loadings", tdata/ttotal*100))
    logger.info("-"*30)
    logger.info(sformat.format("FINAL PROFIT", backtest_results.final_balance) + f" ({backtest_results.ndeals} deals)") 
    logger.info(sformat.format("RECOVRY FACTOR", backtest_results.metrics["linearity"])[:-2]) 
    logger.info(sformat.format("MAXWAIT", backtest_results.metrics["maxwait"])[:-2]+"\n") 
    # import pickle
    # pickle.dump((cfg, broker), open(str(Path("backtests") / f"btest{0:003.0f}.pickle"), "wb"))
    return backtest_results
    
    
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    cfg = PyConfig().test()
    btest_results = backtest(cfg)
    plt.subplots(figsize=(15, 8))
    plt.plot(btest_results.dates, btest_results.balance, linewidth=2, alpha=0.6)
    plt.tight_layout()
    plt.savefig("backtest.png")