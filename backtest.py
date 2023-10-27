import pickle
from pathlib import Path
from shutil import rmtree
from time import time
# import finplot as fplt
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
# import yfinance as yf
from loguru import logger
pd.options.mode.chained_assignment = None
from experts import ExpertFormation, PyConfig
from utils import Broker



class DataParser():
    def __init__(self, cfg):
        self.cfg = cfg     
        
    def load(self):
        p = Path("data") / self.cfg.data_type / self.cfg.period
        flist = [f for f in p.glob("*") if self.cfg.ticker in f.stem]
        if len(flist) == 1:
            return {"metatrader": self.metatrader,
                    "bitfinex":self.bitfinex,
                    }.get(self.cfg.data_type, None)(flist[0])
        else:
            raise FileNotFoundError()
        
    @staticmethod
    def metatrader(data_file):
        pd.options.mode.chained_assignment = None
        hist = pd.read_csv(data_file)
        hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])
        hist.set_index("Date", inplace=True, drop=True)
        hist.drop("Time", axis=1)
        hist["Id"] = list(range(hist.shape[0]))
        return hist
    
    @staticmethod
    def bitfinex(data_file):
        hist = pd.read_csv(data_file, header=1)
        hist = hist.iloc[::-1]
        hist["Date"] = pd.to_datetime(hist.date.values)
        hist.set_index("Date", inplace=True, drop=True)
        hist["Id"] = list(range(hist.shape[0]))
        hist.drop(["unix", "symbol", "date"], axis=1)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Volume"] = hist.iloc[:, -2]
        return hist

def get_data(hist, t, size):
    current_row = hist.iloc[t:t+1].copy()
    current_row.Close.iloc[0] = current_row.Open.iloc[0]
    current_row.High.iloc[0] = current_row.Open.iloc[0]
    current_row.Low.iloc[0] = current_row.Open.iloc[0]
    current_row.Volume[0] = 0
    return pd.concat([hist[t-size-1:t], current_row])

def backtest(cfg):
    exp = ExpertFormation(cfg)
    broker = Broker()
    hist = DataParser(cfg).load()
    save_path = Path("runs") / f"{cfg.ticker}-{cfg.period}"
    if save_path.exists():
        rmtree(save_path)
    save_path.mkdir()
    tstart = max(cfg.hist_buffer_size+1, cfg.tstart)
    tend = cfg.tend if cfg.tend is not None else hist.shape[0]
    t0 = time()
    for t in range(tstart, tend):
        h = get_data(hist, t, cfg.hist_buffer_size)
        if t < tstart or len(broker.active_orders) == 0:
            exp.update(h)
            broker.active_orders = exp.orders
            
        pos = broker.update(h)
        # mpf.plot(hist[t-exp.body_length-1:t], type='candle', alines=dict(alines=exp.lines))
        if pos is not None:
            logger.debug(f"t = {t} -> postprocess closed position")
            broker.close_orders(h.index[-2])
            if cfg.save_plots:
                ords_lines = [ord.lines for ord in broker.orders if ord.open_date in h.index and h.loc[ord.open_date].Id >= pos.open_indx]
                lines2plot = [exp.lines] + ords_lines + [pos.lines]
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if pos.profit > 0 else "red"]
                widths = [1]*len(lines2plot)
                fig = mpf.plot(hist.loc[lines2plot[0][0][0]:lines2plot[-1][-1][0]], 
                            type='candle', 
                            block=False,
                            alines=dict(alines=lines2plot, colors=colors, linewidths=widths),
                            savefig=save_path / f"fig-{pos.open_date}.png")
                del fig
            exp.reset_state()
    # pickle.dump(brok_results, open(str(save_path / f"broker.pickle"), "wb"))
    logger.debug("-"*40 + "\n")
    logger.debug(f"backtest time: {time() - t0:.1f} sec")
    return broker
    
    
if __name__ == "__main__":
    brok_results = backtest(PyConfig().test())
    plt.plot(brok_results.profits.cumsum())
    plt.show()