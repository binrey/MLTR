from typing import List

import finplot as fplt
import pandas as pd

from common.type import Side
from trade.utils import Position


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist2plot = None
    
    def update_hist(self, h):
        if h is None:
            return
        if self.hist2plot is None:
            self.hist2plot = pd.DataFrame(h)
            self.hist2plot["Date"] = pd.to_datetime(self.hist2plot["Date"])
            self.hist2plot.set_index("Date", drop=True, inplace=True)
        else:
            h = pd.DataFrame(h).iloc[-2:]
            h["Date"] = pd.to_datetime(h["Date"])
            h.set_index("Date", drop=True, inplace=True)
            self.hist2plot.iloc[-1, :] = h.iloc[0]
            self.hist2plot = pd.concat([self.hist2plot, h.iloc[1:2, :]])   
             
    def __call__(self, pos_list: List[Position]):
        fplt.candlestick_ochl(self.hist2plot[['Open', 'Close', 'High', 'Low']])
        for pos in pos_list:
            if pos is None:
                continue
            profit = pos.profit
            end_time, end_price = pos.close_date, pos.close_price
            if profit is None:
                profit = pos.cur_profit(self.hist2plot["Open"][-1])
                end_time, end_price = self.hist2plot.index[-1], self.hist2plot["Open"][-1]
                
            rect = fplt.add_rect((end_time, end_price), 
                                 (pos.open_date, pos.open_price), 
                                 color='#8c8' if pos.side == Side.BUY else '#c88')
            
            line = fplt.add_line((pos.open_date, pos.open_price), 
                                 (end_time, end_price), 
                                 color='#000',
                                 width=2, 
                                 style="--")
            
            for t, p in pos.sl_hist:
                fplt.add_line((t - self.cfg.period.to_timedelta(), p), 
                              (t, p), 
                              width=2)
        fplt.winh = 600
        fplt.show()