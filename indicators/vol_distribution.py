import numpy as np


class VolDistribution:
    def __init__(self, cfg):
        self.vol_hist, self.price_bins = None, None
    
    def update(self, h):
        self.price_bins = np.linspace(h["Low"][:-1].min(), h["High"][:-1].max(), 20)
        x = (h["High"][:-1] + h["Low"][:-1])/2
        y = h["Volume"][:-1]
        self.vol_hist = np.histogram(x, bins=self.price_bins, weights=y)[0]
        return [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]