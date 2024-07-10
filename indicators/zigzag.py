import numpy as np
from easydict import EasyDict
from loguru import logger


class ZigZag:
    def __init__(self):
        self.mask, self.min_last, self.max_last, self.last_close, self.last_id = None, None, None, None, 0

    def _get_mask(self, h):
        if self.mask is None or h.Id[-1] - self.last_id > 1:
            self.min_last, self.max_last, self.last_close = h.Low[0], h.High[0], h.Close[0]
            self.mask = np.zeros(h.Id.shape[0] - 1)
            self.mask[0] = 1 if h.High[1] > self.max_last else -1
        else:
            self.mask[:-1] = self.mask[1:]
            self.mask[-1] = 0
        for i in range(2, h.Id.shape[0]):
            if self.mask[i-1] != 0:
                continue
            self.mask[i-1] = self.mask[i-2]
            if h.Close[i] <= self.last_close:
                self.mask[i-1] = -1
            if h.Close[i] >= self.last_close:
                self.mask[i-1] = 1
            self.min_last, self.max_last, self.last_close = h.Low[i], h.High[i], h.Close[i]
        self.last_id = h.Id[-1]
        return self.mask

    def update(self, h):
        self._get_mask(h)
        return self.mask2zigzag(h, self.mask, use_min_max=True)
    
    def mask2zigzag(self, h, mask, use_min_max=False):
        ids, values, types = [], [], []
        def upd_buffers(i, v, t):
            ids.append(i)
            values.append(v)
            types.append(t)
        for i in range(mask.shape[0]):
            if use_min_max:
                y = h.Low[i] if mask[i] > 0 else h.High[i]
            else:
                y = h.Close[i]
            if i > 0: 
                if mask[i] != node:
                    upd_buffers(h.Id[i], y, node)
                    node = mask[i]
            else:
                node = mask[i]
                upd_buffers(h.Id[i], y, -node)
        upd_buffers(h.Id[i]+1, h.Close[-1], mask[i])
        return ids, values, types   


if __name__ == "__main__":
    from backtest import DataParser, MovingWindow
    from experts import PyConfig
    import mplfinance as mpf
    from pathlib import Path
    from shutil import rmtree

    
    cfg = PyConfig().test()
    
    save_path = Path("zz_debug")
    if save_path.exists():
        rmtree(save_path)
    save_path.mkdir()    
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg.hist_buffer_size)
    data_wind, _ = mw(700)
    indc = ZigZagOpt(max_drop=0.01)
    ids, values, types = indc.update(data_wind)
    print(ids, types, values)
    indc.plot(save_path)
    
    hist2plot = hist_pd.iloc[ids[0]:ids[-1]+1]

    lines, thks = [], []
    for m in np.array(indc.masks)[[0, -1]]:
        ids, values, _ = indc.mask2zigzag(data_wind, m, True)
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
    

        
