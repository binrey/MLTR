from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import finplot as fplt
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt

from common.type import Side, TimePeriod
from common.utils import date2str
from trade.utils import Position


@dataclass
class DrawItem:
    line: List[Tuple]
    color: str
    
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
        drawitems4pos: List[DrawItem] = []
        drawitems4sl: List[DrawItem] = []
        for pos in pos_list:
            if pos is None:
                continue
            end_time, end_price = pos.close_date, pos.close_price
            profit = pos.profit
            
            if profit is None:
                profit = pos.cur_profit(self.hist2plot["Open"].iloc[-1])
                end_time, end_price = self.hist2plot.index[-1], self.hist2plot["Open"].iloc[-1]
                
            if pd.to_datetime(end_time) < self.hist2plot.index[0]:
                continue
            drawitem = DrawItem(line=[(pd.to_datetime(pos.open_date.astype("datetime64[m]")), pos.open_price), 
                                      (pd.to_datetime(end_time), end_price)],
                                color='#8c8' if pos.side == Side.BUY else '#c88')
            drawitems4pos.append(drawitem)
            
            for t, p in pos.sl_hist:
                drawitems4sl.append(DrawItem(line=[(pd.to_datetime(t - self.period.to_timedelta()), p), 
                                                   (pd.to_datetime(t), p)], 
                                             color="#000"))
        if self.show:
            self.visualize(drawitems4pos, drawitems4sl)
        if self.save_plots:
            pos_curr_side = pos_list[-1].side if pos_list[-1] else None
            self.save(drawitems4pos, drawitems4sl, pos_curr_side)
        

    def visualize(self, drawitems4possitions: List[DrawItem], drawitems4sl: List[DrawItem]):
        fplt.candlestick_ochl(self.hist2plot[['Open', 'Close', 'High', 'Low']])
        for drawitem in drawitems4possitions:
            rect = fplt.add_rect(drawitem.line[1], drawitem.line[0], color=drawitem.color)
            
            line = fplt.add_line(drawitem.line[0], 
                                 drawitem.line[1], 
                                 color="#000",
                                 width=2, 
                                 style="--")
            
        for drawitem in drawitems4sl:
            fplt.add_line(drawitem.line[0], 
                          drawitem.line[1], 
                          color=drawitem.color,
                          width=2, 
                          style="-")
        fplt.winh = 600
        fplt.show()


    def save(self, drawitems4possitions: List[DrawItem], drawitems4sl: List[DrawItem], side_current: Optional[Side]):
        mystyle = mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
        lines = [drawitem.line for drawitem in drawitems4possitions + drawitems4sl]
        for line in lines:
            for i, point in enumerate(line):
                if point[0] < self.hist2plot.index[0]:
                    line[i] = (self.hist2plot.index[0], point[1])
        
        colors = [drawitem.color for drawitem in drawitems4possitions + drawitems4sl]
        kwargs = dict(
            type='candle',
            block=False,
            alines=dict(alines=lines, colors=colors, linewidths=[1]*len(lines)),
            volume=True,
            figscale=1.5,
            style=mystyle,
            datetime_format='%m-%d %H:%M:%Y',
            # title=f"{np.array(time).astype('datetime64[m]')}-{ticker}-{side.name}",
            returnfig=True
        )

        fig, axlist = mpf.plot(data=self.hist2plot, **kwargs)

        if side_current is not None:
            x, y = self.hist2plot.shape[0]-2, drawitems4possitions[-1].line[0][1]
            id4scale = min(self.hist2plot.shape[0], 10)
            arrow_size = (self.hist2plot.iloc[-id4scale:].High - self.hist2plot.iloc[-id4scale:].Low).mean()
            axlist[0].annotate("", (x, y + arrow_size*side_current.value), fontsize=20, xytext=(x, y),
                        color="black",
                        arrowprops=dict(
                            arrowstyle='->',
                            facecolor='b',
                            edgecolor='b'))

        save_name = date2str(self.hist2plot.index[-1].to_datetime64()) + ".png"
        fig.savefig(self.path2save / save_name, bbox_inches='tight', pad_inches=0.2)
        plt.close('all')