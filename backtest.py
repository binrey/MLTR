import yfinance as yf
import finplot as fplt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import sys
from utils import Broker
from experts import pyconfig, ExpertFormation
from pathlib import Path
from shutil import rmtree
from loguru import logger

cfg = pyconfig()#.read("configs/expert_triangle_simple.yaml")
exp = ExpertFormation(cfg)

pd.options.mode.chained_assignment = None
# data_file = Path("TSLA.scv")
# msft = yf.Ticker("TSLA")
# hist = msft.history(period="70y", interval="1d")
# hist.head()
# print(hist.shape[0])

data_file = Path("data/SBER_M30_200801091100_202309292330.csv")
hist = pd.read_csv(data_file)
hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])
hist.set_index("Date", inplace=True, drop=True)
hist.drop("Time", axis=1)
hist["Id"] = list(range(hist.shape[0]))


period = 20
broker = Broker()
nplot, tstart, tend = 1, 100, hist.shape[0] - 500
#fig = mpf.figure(figsize=(10, 10))
save_path = Path(f"runs/{data_file.stem}")
if save_path.exists():
    rmtree(save_path)
save_path.mkdir()

def get_data(hist, t, size):
    current_row = hist.iloc[t:t+1].copy()
    current_row.Close[0] = current_row.Open.values[0]
    current_row.High[0] = current_row.Open[0]
    current_row.Low[0] = current_row.Open[0]
    current_row.Volume[0] = 0
    return pd.concat([hist[t-size-1:t], current_row])

for t in range(tstart, tend):
    h = get_data(hist, t, period)
    if t < tstart or len(broker.active_orders) == 0:
        exp.update(h)
        broker.active_orders = exp.orders
        
    pos = broker.update(h)
    # mpf.plot(hist[t-exp.body_length-1:t], type='candle', alines=dict(alines=exp.lines))
    if pos is not None:
        logger.debug(f"t = {t} -> postprocess closed position")
        broker.close_orders(h.index[-2])
        ords_lines = [ord.lines for ord in broker.orders if ord.open_date in h.index and h.loc[ord.open_date].Id >= pos.open_indx]
        lines2plot = [exp.lines] + ords_lines + [pos.lines]
        colors = ["blue"]*(len(lines2plot)-1) + ["green" if pos.profit > 0 else "red"]
        widths = [1]*len(lines2plot)
        # fig = mpf.plot(hist.loc[lines2plot[0][0][0]:lines2plot[-1][-1][0]], 
        #             type='candle', 
        #             block=False,
        #             alines=dict(alines=lines2plot, colors=colors, linewidths=widths),
        #             savefig=save_path / f"fig-{pos.open_date}.png")
        # del fig
        exp.reset_state()



import numpy as np
import pickle
pickle.dump(broker, open(str(save_path / f"broker.pickle"), "wb"))
plt.plot(broker.profits.cumsum())
plt.show()