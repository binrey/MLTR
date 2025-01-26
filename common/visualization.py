from pathlib import Path
from typing import List, Optional, Union

import finplot as fplt
import matplotlib
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from experts.core.expert import ExpertBase

matplotlib.use('agg')

from common.type import Line, Side, TimePeriod, TimeVolumeProfile
from common.utils import date2str
from trade.utils import Position


class Visualizer:
    def __init__(self, 
                 period: TimePeriod, 
                 show: bool, 
                 save_to: Optional[str], 
                 vis_hist_length: int) -> None:
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
            # self.hist2plot.iloc[-1, :] = h.iloc[0]
            self.hist2plot = pd.concat([self.hist2plot.iloc[:-1, :], h])  
            # if self.hist2plot.shape[0] > self.vis_hist_length:
            self.hist2plot = self.hist2plot.iloc[-self.vis_hist_length:] 
             
    def __call__(self, pos_list: List[Position], expert2draw: Optional[ExpertBase] = None):
        if not self.show and not self.save_plots:
            return
        drawitems4pos: List[Line] = []
        drawitems4sl: List[Line] = []
        drawitems4tp: List[Line] = []
        for pos in pos_list:
            if pos is None:
                continue
            end_time, end_price = pos.close_date, pos.close_price
            profit = pos.profit
            
            if profit is None or end_time is None:
                profit = pos.cur_profit(self.hist2plot["Open"].iloc[-1])
                end_time, end_price = self.hist2plot.index[-1], self.hist2plot["Open"].iloc[-1]
            else:
                end_time = end_time.astype("datetime64[m]")
                
            if pd.to_datetime(end_time) < self.hist2plot.index[0]:
                continue
            drawitem = Line(line=[(pd.to_datetime(pos.open_date.astype("datetime64[m]")), pos.open_price), 
                                      (pd.to_datetime(end_time), end_price)],
                                color='#8c8' if pos.side == Side.BUY else '#c88')
            drawitems4pos.append(drawitem)
            
            for t, p in pos.sl_hist:
                drawitems4sl.append(Line(line=[(pd.to_datetime(t - self.period.to_timedelta()), p), 
                                                   (pd.to_datetime(t), p)], 
                                             color="#000"))

            for t, p in pos.tp_hist:
                drawitems4tp.append(Line(line=[(pd.to_datetime(t - self.period.to_timedelta()), p), 
                                                   (pd.to_datetime(t), p)], 
                                             color="#000"))
                
        if expert2draw is not None:
            drawitems4expert = expert2draw.decision_maker.vis_objects
        else:
            drawitems4expert = []
                
        if self.show:
            self.visualize(drawitems4pos, drawitems4sl, drawitems4tp, drawitems4expert)
            return None
        if self.save_plots:
            pos_curr_side = None
            if len(pos_list) and pos_list[-1] is not None:
                pos_curr_side = pos_list[-1].side
            return self.save(drawitems4pos, drawitems4sl, drawitems4tp, pos_curr_side)
        

    def visualize(self, 
                  drawitems4possitions: List[Line], 
                  drawitems4sl: List[Line], 
                  drawitems4tp: List[Line], 
                  drawitems4expert: List[Union[Line, TimeVolumeProfile]]) -> None:
        ax = fplt.create_plot('long term analysis', rows=1, maximize=False)
        fplt.candlestick_ochl(self.hist2plot[['Open', 'Close', 'High', 'Low']])
        fplt.volume_ocv(self.hist2plot[['Open', 'Close', 'Volume']], ax=ax.overlay(scale=0.08))
        for drawitem in drawitems4expert:
            if type(drawitem) is TimeVolumeProfile:
                time_vol_profile = [[drawitem.time if drawitem.time in self.hist2plot.index else self.hist2plot.index[0], drawitem.hist],
                                    [self.hist2plot.index[-1], [(1, 1)]]]
                fplt.horiz_time_volume(time_vol_profile, draw_va=0, draw_poc=3.0)
        
        for drawitem in drawitems4possitions:
            rect = fplt.add_rect(drawitem.line[1], drawitem.line[0], color=drawitem.color)
            
            line = fplt.add_line(drawitem.line[0], 
                                 drawitem.line[1], 
                                 color="#000",
                                 width=2, 
                                 style="--")
            
        for drawitem in drawitems4sl + drawitems4tp:
            fplt.add_line(drawitem.line[0], 
                          drawitem.line[1], 
                          color=drawitem.color,
                          width=2, 
                          style="-")
            
        for drawitem in drawitems4expert:
            if type(drawitem) is Line:
                for i in range(2):
                    if drawitem.line[i][0] is None:
                        drawitem.line[i] = (self.hist2plot.index[-i], drawitem.line[i][1])
                fplt.add_line(drawitem.line[0], 
                            drawitem.line[1], 
                            color=drawitem.color,
                            width=1, 
                            style="--")
        fplt.winh = 600
        fplt.show()


    def save(self, drawitems4possitions: List[Line], drawitems4sl: List[Line], drawitems4tp: List[Line], side_current: Optional[Side]):
        try:
            mystyle = mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
            lines = [drawitem.line for drawitem in drawitems4possitions + drawitems4sl + drawitems4tp]
            for line in lines:
                for i, point in enumerate(line):
                    if point[0] < self.hist2plot.index[0]:
                        line[i] = (self.hist2plot.index[0], point[1])
                    if point[0] > self.hist2plot.index[-1]:
                        line[i] = (self.hist2plot.index[-1], point[1])                    
            
            colors = [drawitem.color for drawitem in drawitems4possitions + drawitems4sl + drawitems4tp]
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

            save_name = (self.path2save / date2str(self.hist2plot.index[-1].to_datetime64(), "m")).with_suffix(".png")
            fig.savefig(save_name, bbox_inches='tight', pad_inches=0.2)
            plt.close('all')
        except Exception as e:
            logger.error(f"error in save plot: {e}")

        return save_name