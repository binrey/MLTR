import sys
from pathlib import Path
from shutil import rmtree
from time import time

import mplfinance as mpf
import numpy as np
import pandas as pd

# sys.path.append(str(Path(__file__).parent.parent))
from common.utils import PyConfig
from common.visualization import Visualizer
from data_processing.dataloading import DataParser, MovingWindow
from indicators import *


class IndcTester():
    def __init__(self, indc, save_path=Path("indicators/test")):
        self.save_path = save_path
        self.indc = indc
        cfg = PyConfig("configs/test_indicator.py").get_inference()
        self.mw = MovingWindow(cfg)
        
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
        
        self.visualizer = Visualizer(period=cfg['period'], 
                                show=cfg['visualize'], 
                                save_to=self.save_path if cfg['save_plots'] else None,
                                vis_hist_length=cfg['vis_hist_length'])  
       
    def test_zigzag_opt(self, t):
        data_wind, _ = self.mw(t)
        ids, values, types = self.indc.update(data_wind)
        self.indc.plot(self.save_path)
        
        hist2plot = self.hist_pd.iloc[ids[0]:ids[-1]+1]

        lines, thks = [], []
        for m in np.array(self.indc.masks)[[0, -1]]:
            ids, values, _ = self.indc.mask2zigzag(data_wind, m, True)
            line = []
            for t, y in zip(ids, values):
                try:
                    y = y.item()
                except:
                    pass
                line.append((hist2plot.index[hist2plot.Id==t][0], y))
            lines.append(line)
            
        thks = list(range(1, len(lines)+1))[::-1]
        colors = [(i*0.2, i*0.2, i*0.2) for i in thks]
        fig = mpf.plot(hist2plot, 
                    type='candle', 
                    block=False,
                    alines=dict(alines=lines, colors=colors, linewidths=thks),
                    # savefig=save_path / f"_final.jpg"
                    )
        mpf.show()    
    

    def test_zigzag(self, t):
        h, _ = self.mw[t]
        ids, values, types = self.indc.update(h)
        
        self.visualizer.update_hist(h)
        
        self.visualizer([])
        
        
    def test_hvol(self, t):
        h, _ = self.mw[t]
        self.visualizer.update_hist(h)
        hist = self.indc.update(h)
        hist = [[self.visualizer.hist2plot.index[0], hist],
                [self.visualizer.hist2plot.index[-1], np.ones_like(hist)]]
        
        self.visualizer([], hist)