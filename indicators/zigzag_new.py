import numpy as np
from easydict import EasyDict
from loguru import logger


class ZigZagNew:
    def __init__(self, period, size=16):
        self.period = period
        self.mask, self.min_last, self.max_last, self.last_close = None, None, None, None
        self.last_id = 0
        self.size = size
        
        self.ids = np.zeros(size, dtype=int)
        self.values = np.zeros(size, dtype=np.float32)
        self.types = np.zeros(size, dtype=int)
        self.n = 0

    def _get_mask(self, h):
        self.mask = None
        if self.mask is None or h.Id[-1] - self.last_id > 1:
            self.min_last = h.Low[:self.period].min()
            self.max_last = h.High[:self.period].max()  
            self.last_extr_ind = 0   
            self.mask = np.zeros(h.Id.shape[0] - 1)
        else:
            self.mask[:-1] = self.mask[1:]
            self.mask[-1] = 0
            
        for i in range(self.period, h.Id.shape[0]-1):
            if self.mask[i] != 0:
                continue
            self.mask[i] = self.mask[i-1]
            if h.Low[i] <= self.min_last:
                self.mask[self.last_extr_ind:i+1] = -1
                self.last_extr_ind = i
            if h.High[i] >= self.max_last:
                self.mask[self.last_extr_ind:i+1] = 1
                self.last_extr_ind = i
            self.min_last = h.Low[i-self.period+1:i+1].min()
            self.max_last = h.High[i-self.period+1:i+1].max()        
        self.last_id = h.Id[-1]
        return self.mask

    def update(self, h):
        self._get_mask(h)
        return self.mask2zigzag(h, self.mask, use_min_max=True)

    def upd_buffers(self, i, v, t):
        self.ids[self.n] = i
        self.values[self.n] = v
        self.types[self.n] = t
        self.n += 1
        
    def mask2zigzag(self, h, mask, use_min_max=False):        
        self.ids[:] = 0
        self.values[:] = 0
        self.types[:] = 0
        self.n = 0
        for i in range(mask.shape[0]):
            if use_min_max:
                y = h.Low[i] if mask[i] > 0 else h.High[i]
            else:
                y = h.Close[i]
            if i > 0: 
                if mask[i] != node:
                    self.upd_buffers(h.Id[i], y, node)
                    node = mask[i]
            else:
                node = mask[i]
                self.upd_buffers(h.Id[i], y, -node)
            if self.n == self.size - 1:
                break
        self.upd_buffers(h.Id[i]+1, h.Close[-1], mask[i])
        return self.ids[self.types!=0], self.values[self.types!=0], self.types[self.types!=0] 

    

        
