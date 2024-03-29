import numpy as np
from easydict import EasyDict
from loguru import logger

class ZigZag:
    def __init__(self):
        self.mask, self.min_last, self.max_last, self.last_id = None, None, None, 0

    def _get_mask(self, h):
        if self.mask is None or h.Id[-1] - self.last_id > 1:
            self.min_last, self.max_last = h.Low[0], h.High[0]
            self.mask = np.zeros(h.Id.shape[0] - 1)
            self.mask[0] = 1 if h.High[1] > self.max_last else -1
        else:
            self.mask[:-1] = self.mask[1:]
            self.mask[-1] = 0
        for i in range(2, h.Id.shape[0]):
            if self.mask[i-1] != 0:
                continue
            self.mask[i-1] = self.mask[i-2]
            if h.Low[i] <= self.min_last and h.High[i] <= self.max_last:
                self.mask[i-1] = -1
                if h.High[i-2] >= h.High[i-1]:
                    self.mask[i-2] = -1
            if h.High[i] >= self.max_last and h.Low[i] >= self.min_last:
                self.mask[i-1] = 1
                if h.Low[i-2] <= h.Low[i-1]:
                    self.mask[i-2] = 1
            self.min_last, self.max_last = h.Low[i], h.High[i]
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
        upd_buffers(h.Id[i]+1, h.Low[-1] if - mask[i] > 0 else h.High[-1], mask[i])
        return ids, values, types   
    

class ZigZag2:
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


class ZigZagOpt(ZigZag2):
    def __init__(self, min_nodes=1, max_drop=1, simp_while_grow=False):
        super(ZigZagOpt, self).__init__()
        self.min_nodes = min_nodes
        self.simp_while_grow = simp_while_grow
        self.max_drop = max_drop
        self.masks = []
    
    def zigzag_simplify(self, data, mask, only_calc=False):
        def swap_node(data, node):
            data[node.id1:node.id2+1] = -data[node.id1:node.id2+1]
            return data
        smax, nnodes = 0, 1
        node = EasyDict({"value": mask[0], "id1": 0, "id2": 0, "metric":None})
        nodemax = node
        for i in range(1, mask.shape[0]):
            if mask[i] == node.value:
                node.id2 = i
            else:
                if not only_calc:
                    mask = swap_node(mask, node)
                    s = (data*mask).sum()
                    if s >= smax:
                        # logger.debug(f"{i}, {nnodes}, {node}, {s:+5.3f}")
                        smax = s
                        nodemax = node
                        nodemax.metric = s
                        # logger.debug("OK")
                    mask = swap_node(mask, node)
                    # print("")
                node = EasyDict({"value": mask[i], "id1": i, "id2": i, "metric":None})
                nnodes += 1
        if not only_calc:
            mask = swap_node(mask, nodemax)
            nnodes -= 1
        else:
            return mask, nnodes, (data*mask).sum()
        if nodemax.metric is None:
            nodemax.metric = (data*mask).sum()
        return mask, nnodes, nodemax.metric


    def update(self, h):
        hclose = h.Close
        dh = hclose[1:] - hclose[:-1]
        mask = self._get_mask(h)
        i, v, t = self.mask2zigzag(h, mask, True) 
        _, nnodes, res = self.zigzag_simplify(dh, mask, only_calc=True)
        logger.debug(f"{nnodes:003}, {res:>7.2f}, {0}")    
        self.res_list = [res]
        self.nnodes_list = [nnodes]
        res_max = res
        while nnodes > self.min_nodes:
            mask_, nnodes, res = self.zigzag_simplify(dh, mask.copy())
            drop = (res_max - res)/max(1, res_max)
            logger.debug(f"{nnodes:003}, {res:>7.2f}, {drop}")    
                
            
            if res < self.res_list[-1]:
                if self.simp_while_grow:
                    break
            else:
                res_max = res
            if drop > self.max_drop:
                break
            self.res_list.append(res)
            self.nnodes_list.append(nnodes)
            mask = mask_.copy()
            self.masks.append(mask)
        return self.mask2zigzag(h, mask, True)
    
    def plot(self, save_path):
        import matplotlib.pylab as plt
        plt.plot(self.nnodes_list, self.res_list, ".-")
        plt.savefig(save_path / "zz_opt.jpg")


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
    

        
