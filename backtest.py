from pathlib import Path
from shutil import rmtree
from time import perf_counter
from datetime import datetime
from easydict import EasyDict
# import finplot as fplt
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
# import yfinance as yf
from loguru import logger
from tqdm import tqdm

pd.options.mode.chained_assignment = None
from experts import ExpertFormation, PyConfig
from utils import Broker, trailing_stop

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

class DataParser():
    def __init__(self, cfg):
        self.cfg = cfg     
        
    def load(self):
        p = Path("data") / self.cfg.data_type / self.cfg.period
        flist = [f for f in p.glob("*") if self.cfg.ticker in f.stem]
        if len(flist) == 1:
            return {"metatrader": self.metatrader,
                    "FORTS": self.metatrader,
                    "bitfinex": self.bitfinex,
                    "yahoo": self.yahoo,
                    }.get(self.cfg.data_type, None)(flist[0])
        else:
            raise FileNotFoundError(p)
        
    @staticmethod
    def metatrader(data_file):
        pd.options.mode.chained_assignment = None
        hist = pd.read_csv(data_file, sep="\t")
        hist.columns = map(lambda x:x[1:-1], hist.columns)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])#, utc=True)
        hist["timestamp"] = [dt.timestamp() for dt in hist.Date]
        hist.set_index("Date", inplace=True, drop=True)
        hist.drop("Time", axis=1, inplace=True)
        columns = list(hist.columns)
        columns[-2] = "Volume"
        hist.columns = columns
        hist["Id"] = list(range(hist.shape[0]))
        hist_dict = EasyDict({c:hist[c].values.squeeze() for c in hist.columns})
        return hist, hist_dict
    
    @staticmethod
    def bitfinex(data_file):
        hist = pd.read_csv(data_file, header=1)
        hist = hist[::-1]
        hist["Date"] = pd.to_datetime(hist.date.values)
        hist.set_index("Date", inplace=True, drop=False)
        hist["Id"] = list(range(hist.shape[0]))
        hist.drop(["unix", "symbol", "date"], axis=1, inplace=True)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Volume"] = hist.iloc[:, -3]
        hist_dict = EasyDict({c:hist[c].values.squeeze() for c in hist.columns})
        return hist, hist_dict
    
    @staticmethod
    def yahoo(data_file):
        hist = pd.read_csv(data_file)
        hist["Date"] = pd.to_datetime(hist.Date, utc=True)
        hist.set_index("Date", inplace=True, drop=True)
        hist["Id"] = list(range(hist.shape[0]))
        return hist
        

class MovingWindow():
    def __init__(self, hist, size):
        self.hist = hist
        self.size = size
        self.data = EasyDict(Id=np.zeros(size, dtype=np.int64),
                             Open=np.zeros(size, dtype=np.float32),
                             Close=np.zeros(size, dtype=np.float32),
                             High=np.zeros(size, dtype=np.float32),
                             Low=np.zeros(size, dtype=np.float32),
                             Volume=np.zeros(size, dtype=np.int64)
                             )
        
    def __call__(self, t):
        t0 = perf_counter()
        self.data.Id = self.hist.Id[t-self.size+1:t+1]
        self.data.Open = self.hist.Open[t-self.size+1:t+1]
        self.data.Close[:-1] = self.hist.Close[t-self.size+1:t]
        self.data.Close[-1] = self.data.Open[-1]
        self.data.High[:-1] = self.hist.High[t-self.size+1:t]
        self.data.High[-1] = self.data.Open[-1]
        self.data.Low[:-1] = self.hist.Low[t-self.size+1:t]
        self.data.Low[-1] = self.data.Open[-1]
        self.data.Volume[:-1] = self.hist.Volume[t-self.size+1:t]
        self.data.Volume[-1] = 0      
        return self.data, perf_counter() - t0


def backtest(cfg):
    exp = ExpertFormation(cfg)
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
    for t in tqdm(range(tstart, tend)):
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
                lines2plot = [exp.lines] + ords_lines + [pos.lines]
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
                            savefig=save_path / f"fig-{pos.open_date}.png")
                del fig
            exp.reset_state()
    
    ttotal = perf_counter() - t0
    logger.info("-"*40)
    sformat = "{:>40}: {:>3.0f} %"
    logger.info("{:>40}: {:.1f} sec".format("total backtest", ttotal))
    logger.info(sformat.format("expert updates", texp/ttotal*100))
    logger.info(sformat.format("broker updates", tbrok/ttotal*100))
    logger.info(sformat.format("data loadings", tdata/ttotal*100))

    return broker
    
    
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    brok_results = backtest(PyConfig().test())
    plt.plot([pos.close_date for pos in brok_results.positions], brok_results.profits.cumsum())
    plt.savefig("backtest.png")
    # plt.show()