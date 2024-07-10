import sys
import mplfinance as mpf
from pathlib import Path
from shutil import rmtree
import numpy as np
from pathlib import Path
from time import time

sys.path.append(str(Path(__file__).parent.parent))
from backtest import DataParser, MovingWindow
from utils import PyConfig
from indicators import *

    
class IndcTester():
    def __init__(self, indc, save_path=Path("indicators/test")):
        self.save_path = save_path
        self.indc = indc
        cfg = PyConfig("test.py").test()
        self.hist_pd, self.hist = DataParser(cfg).load()
        self.mw = MovingWindow(self.hist, cfg.hist_buffer_size)
        
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
       
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
        data_wind, _ = self.mw(t)
        ids, values, types = self.indc.update(data_wind)
        
        hist2plot = self.hist_pd.iloc[ids[0]:ids[-1]+1]

        line = []
        for t, y in zip(ids, values):
            try:
                y = y.item()
            except:
                pass
            line.append((hist2plot.index[hist2plot.Id==t][0], y))
            
        fig = mpf.plot(hist2plot, 
                    type='candle', 
                    block=False,
                    alines=dict(alines=[line]),
                    savefig=self.save_path / f"zz_{t}.jpg"
                    )
        mpf.show()    
    
    
if __name__ == "__main__":
    indc = ZigZagNew(8)
    tester = IndcTester(indc)
        
    t0 = time()
    for i, t in enumerate(range(990, 1050)):
        tester.test_zigzag(t)
    print((time()-t0)/(i+1))
    
    

