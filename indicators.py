import numpy as np
from easydict import EasyDict
from loguru import logger
# import numba
# from numba import jit, njit


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
            if h.Low[i] < self.min_last and h.High[i] < self.max_last:
                self.mask[i-1] = -1
                if h.High[i-2] > h.High[i-1]:
                    self.mask[i-2] = -1
            if h.High[i] > self.max_last and h.Low[i] > self.min_last:
                self.mask[i-1] = 1
                if h.Low[i-2] < h.Low[i-1]:
                    self.mask[i-2] = 1
            self.min_last, self.max_last = h.Low[i], h.High[i]
        self.last_id = h.Id[-1]
        return self.mask

    # def _get_mask2(self, h):
    #     if self.mask is None or h.Id[-1] - self.last_id > 1:
    #         self.min_last, self.max_last = h.Low[0], h.High[0]
    #         self.mask = np.zeros(h.shape[0] - 1, dtype=np.int64)
    #         self.mask[0] = 1 if h.High[1] > self.max_last else -1
    #     else:
    #         self.mask[:-1] = self.mask[1:]
    #         self.mask[-1] = 0
    #     hlow = h.Low.values
    #     htop = h.High.values
    #     hsize = h.shape[0]            
    #     self.mask, self.max_last, self.min_last = self._zloop(self.mask, hlow, htop, hsize, self.max_last, self.min_last)
    #     self.last_id = h.Id[-1]
    #     return self.mask

    # @staticmethod
    # @jit(nopython=True)
    # def _zloop(mask, hlow, htop, hsize, max_last, min_last):
    #     for i in range(2, hsize):
    #         if mask[i-1] != 0:
    #             continue
    #         mask[i-1] = mask[i-2]
    #         if hlow[i] < min_last and htop[i] < max_last:
    #             mask[i-1] = -1
    #             if htop[i-2] > htop[i-1]:
    #                 mask[i-2] = -1
    #         if htop[i] > max_last and hlow[i] > min_last:
    #             mask[i-1] = 1
    #             if hlow[i-2] < hlow[i-1]:
    #                 mask[i-2] = 1
    #         min_last, max_last = hlow[i], htop[i]     
    #     return mask, max_last, min_last

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
                y = h.Low[i] if mask[i] > 0 else h.High[i]
            else:
                y = h.Close[i]
            if i > 0: 
                if mask[i] != node:
                    upd_buffers(h.Id[i], x, y, node)
                    node = mask[i]
            else:
                node = mask[i]
                upd_buffers(h.Id[i], x, y, mask[i])
        upd_buffers(i+1, h.index[-1], h.Low[-1] if - mask[i] > 0 else h.High[-1], mask[i])
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


if __name__ == "__main__":
    from time import perf_counter
    # @jit(nopython=True)
    def z(mask, hlow, htop, hsize, max_last, min_last):
        for i in range(2, hsize):
            # if mask[i-1] != 0:
            #     continue
            mask[i-1] = mask[i-2]
            if hlow[i] < min_last and htop[i] < max_last:
                mask[i-1] = -1
                if htop[i-2] > htop[i-1]:
                    mask[i-2] = -1
            if htop[i] > max_last and hlow[i] > min_last:
                mask[i-1] = 1
                if hlow[i-2] < hlow[i-1]:
                    mask[i-2] = 1
            min_last, max_last = hlow[i], htop[i]     
        return mask, max_last, min_last
    
    
    mask = np.load("mask.npy").astype(np.float32)
    hlow = np.load("hlow.npy")
    htop = np.load("htop.npy")
    hsize = 32
    max_last = np.load("max_last.npy").item()
    min_last = np.load("min_last.npy").item()
    
    t_tot, t_sum = 0, 0
    t0 = perf_counter()
    for i in range(100000):
        tt0 = perf_counter()
        output = z(mask, hlow, htop, hsize, max_last, min_last)
        t_sum += perf_counter() - tt0
    print(perf_counter() - t0, t_sum)
        
