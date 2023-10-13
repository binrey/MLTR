import yfinance as yf
import finplot as fplt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import sys
from utils import Triangle, Trend, Broker
from pathlib import Path
from shutil import rmtree
from loguru import logger

# data_file = Path("TSLA.scv")
# msft = yf.Ticker("TSLA")
# hist = msft.history(period="70y", interval="1d")
# hist.head()
# print(hist.shape[0])

data_file = Path("data/SBER_M30_200801091100_202309292330.csv")
hist = pd.read_csv(data_file)
hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])
hist.set_index("Date", inplace=True, drop=True)



broker = Broker()
nplot, tstart, tend = 1, 100, hist.shape[0] - 500
#fig = mpf.figure(figsize=(10, 10))
save_path = Path(f"runs/{data_file.stem}")
if save_path.exists():
    rmtree(save_path)
save_path.mkdir()
CFig = Trend
candlefig = CFig(60, 3, 2, 1)
for t in range(tstart, tend):
    if t > tstart and len(broker.active_orders) > 0:
        pos = broker.update(hist[:t])
        # mpf.plot(hist[t-candlefig.body_length-1:t], type='candle', alines=dict(alines=candlefig.lines))
        if pos is not None:
            logger.debug(f"t = {t} -> postprocess closed position")
            broker.close_orders()
            # fig = mpf.figure(figsize=(16, 10))
            # ax = fig.add_subplot(2,3,nplot) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
            
            # candlefig.lines.append([(pos.open_date, pos.open_price), (pos.close_date, pos.close_price)])
            colors = ["blue"]*(len(candlefig.lines)-1) + ["green" if pos.profit > 0 else "red"]
            widths = [1]*len(candlefig.lines)
            dt = pos.close_indx-pos.open_indx
            try:
                fig = mpf.plot(hist.loc[candlefig.lines[0][0][0]:hist.index[t+1]], 
                            type='candle', 
                            block=False,
                            alines=dict(alines=candlefig.lines, colors=colors, linewidths=widths),
                            savefig=save_path / f"fig-{pos.open_date}.png")
            except Exception as ex:
                logger.error(f"{t} -> {ex}")
            finally:
                del fig
            candlefig = CFig(60, 3, 2, 1)
                          
    else:
        candlefig.update(hist, t)
        broker.active_orders = candlefig.orders


import numpy as np
import pickle
pickle.dump(broker, open(str(save_path / f"broker.pickle"), "wb"))
plt.plot(broker.profits.cumsum())
plt.show()