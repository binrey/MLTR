import yfinance as yf
import finplot as fplt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from utils import Triangle as CFig, Broker
from pathlib import Path
from shutil import rmtree

# msft = yf.Ticker("MSFT")
# hist = msft.history(period="70y", interval="1d")
# hist.head()
# print(hist.shape[0])

data_file = Path("data/GAZP_M30_201301081000_202309292330.csv")
hist = pd.read_csv(data_file)
hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])
hist.set_index("Date", inplace=True, drop=True)

targets, predicts = [], []

broker = Broker()
nplot, tstart = 1, 100
#fig = mpf.figure(figsize=(10, 10))
save_path = Path(f"runs/{data_file.stem}")
if save_path.exists():
    rmtree(save_path)
save_path.mkdir()
candlefig = CFig(30, 30, 1, 1)
for t in range(tstart, 55000):
    if t > tstart and len(broker.active_orders) > 0:
        pos = broker.update(hist[:t])
        # mpf.plot(hist[t-candlefig.body_length-1:t], type='candle', alines=dict(alines=candlefig.lines))
        if pos is not None:
            broker.close_orders()
            fig = mpf.figure(figsize=(16, 10))
            # ax = fig.add_subplot(2,3,nplot) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
            
            candlefig.target_line = [(pos.open_date, pos.open_price), (pos.close_date, pos.close_price)]
            colors = ["blue"]*(len(candlefig.lines)-1) + ["green" if pos.profit > 0 else "red"]
            widths = [1]*len(candlefig.lines)
            dt = pos.close_indx-pos.open_indx
            mpf.plot(hist[t-dt-candlefig.body_length-10:t+10], 
                        type='candle', 
                    #  ax=ax, 
                        alines=dict(alines=candlefig.lines, colors=colors, linewidths=widths),
                        savefig=save_path / f"fig-{pos.open_date}.png")
                
            candlefig = CFig(30, 30, 1, 1)
                          
    else:
        candlefig.update(hist, t)
        broker.active_orders = candlefig.orders

        
print(len(targets))
mpf.show()


import numpy as np
plt.plot(np.array(broker.profits).cumsum())
plt.show()