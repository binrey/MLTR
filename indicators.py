import numpy as np
from easydict import EasyDict
from loguru import logger

class ZigZag:
    def __init__(self):
        self.mask, self.min_last, self.max_last = None, None, None
        
    def _get_mask(self, h):
        if self.mask is None:
            self.min_last, self.max_last = h.Low[0], h.High[0]
            self.mask = np.zeros(h.shape[0] - 1)
            self.mask[0] = 1 if h.High[1] > self.max_last else -1
        else:
            size2add = max(0, h.shape[0] - self.mask.shape[0] - 1)
            self.mask = np.append(self.mask, np.zeros((size2add,)))
        for i in range(2, h.shape[0]):
            if self.mask[i-1] != 0:
                continue
            self.mask[i-1] = self.mask[i-2]
            if h.Low[i] < self.min_last and h.High[i] < self.max_last:
                self.mask[i-1] = -1
                if h.High[i-2] > h.High[i-1]:
                    self.mask[i-2] = -1
            if h.High[i] > self.max_last and h.Low[i] > self.min_last:
                self.mask[i-1] = 1
                if h.Low[i-2] < h.Low[i-1]:
                    self.mask[i-2] = 1
            self.min_last, self.max_last = h.Low[i], h.High[i]
        return self.mask

    def update(self, h):
        self._get_mask(h)
        return self.mask2zigzag(h, self.mask, use_min_max=True)

    def mask2zigzag(self, h, mask, use_min_max=False):
        ids, dates, values, types = [], [], [], []
        def upd_buffers(i, d, v, t):
            ids.append(i)
            dates.append(d)
            values.append(v)
            types.append(t)
        for i in range(mask.shape[0]):
            x = h.index[i]
            if use_min_max:
                y = h.Low.values[i] if mask[i] > 0 else h.High.values[i]
            else:
                y = h.Close.values[i]
            if i > 0: 
                if mask[i] != node:
                    upd_buffers(i, x, y, node)
                    node = mask[i]
            else:
                node = mask[i]
                upd_buffers(i, x, y, mask[i])
        upd_buffers(i+1, h.index[-1], h.Low.values[-1] if - mask[i] > 0 else h.High.values[-1], mask[i])
        return ids, dates, values, types   
    
    
def zigzag_simplify(data, mask, only_calc=False):
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
                # print(f"{i}, {nnodes}, {node}, {s:+5.3f}", end=" ")
                if s >= smax:
                    smax = s
                    nodemax = node
                    nodemax.metric = s
                    # print("OK")
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


def zz_opt(h, minnodes=1, simp_while_grow=True):
    hclose = h.Close.values
    dh = hclose[1:] - hclose[:-1]
    zz = ZigZag()
    mask = zz._get_mask(h)
    i, d, v, t = zz.mask2zigzag(h, mask, True) 
    _, nnodes, res = zigzag_simplify(dh, mask, only_calc=True)
    res_list = [res]
    nnodes_list = [nnodes]
    while nnodes > minnodes:
        mask, nnodes, res = zigzag_simplify(dh, mask)
        if simp_while_grow:
            if res < res_list[-1]:
                return zz.mask2zigzag(h, mask, True)
        res_list.append(res)
        nnodes_list.append(nnodes)
        # i, d, v, t = zz.mask2zigzag(h, mask, True)
        # lines = [(x, y) for x, y in zip(d, v)]
        # mpf.plot(h, type="candle", alines=lines, title=str(nnodes))   
    return zz.mask2zigzag(h, mask, True)