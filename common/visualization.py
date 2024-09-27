from pathlib import Path
from typing import List, Optional

import finplot as fplt
import pandas as pd

from common.type import Side, TimePeriod
from common.utils import date2str
from trade.utils import Position


class Visualizer:
    def __init__(self, period: TimePeriod, show: bool, save_to: Optional[str], vis_hist_length: int) -> None:
        self.period = period
        self.show = show
        self.save_plots = True if save_to is not None else False
        self.vis_hist_length = vis_hist_length
        
        if self.save_plots:
            self.path2save = Path(save_to)
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
            # if self.hist2plot.shape[0] > self.vis_hist_length:
            self.hist2plot = self.hist2plot.iloc[-self.vis_hist_length:] 
             
    def __call__(self, pos_list: List[Position]):
        if not self.show and not self.save_plots:
            return
        fplt.candlestick_ochl(self.hist2plot[['Open', 'Close', 'High', 'Low']])
        for pos in pos_list:
            if pos is None:
                continue
            end_time, end_price = pos.close_date, pos.close_price
            profit = pos.profit
            
            if profit is None:
                profit = pos.cur_profit(self.hist2plot["Open"][-1])
                end_time, end_price = self.hist2plot.index[-1], self.hist2plot["Open"][-1]
                
            if pd.to_datetime(end_time) < self.hist2plot.index[0]:
                continue
            
            rect = fplt.add_rect((end_time, end_price), 
                                 (pos.open_date.astype("datetime64[m]"), pos.open_price), 
                                 color='#8c8' if pos.side == Side.BUY else '#c88')
            
            line = fplt.add_line((pos.open_date.astype("datetime64[m]"), pos.open_price), 
                                 (end_time, end_price), 
                                 color='#000',
                                 width=2, 
                                 style="--")
            
            for t, p in pos.sl_hist:
                fplt.add_line((t - self.period.to_timedelta(), p), 
                              (t, p), 
                              width=2)
        fplt.winh = 600
        fplt.timer_callback(update_func=self.save_func,
                            seconds=0.5,
                            single_shot=True)
        fplt.show()
                
    def save_func(self):
        if self.save_plots:
            save_name = date2str(self.hist2plot.index[-1].to_datetime64()) + ".png"
            fplt.screenshot(open(self.path2save / save_name, 'wb'))
            if not self.show:
                fplt.close()