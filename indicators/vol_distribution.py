import numpy as np
import pandas as pd

from common.type import TimeVolumeProfile


class VolDistribution:
    def __init__(self, nbins=20):
        self.nbins = nbins
        self.vol_hist, self.price_bins = None, None
        self.vis_objects = []
    
    def update(self, h) -> TimeVolumeProfile:
        self.price_bins = np.linspace(h["Low"][:-1].min(), h["High"][:-1].max(), self.nbins)
        x = (h["High"][:-1] + h["Low"][:-1])/2
        y = h["Volume"][:-1]
        self.vol_hist = np.histogram(x, bins=self.price_bins, weights=y)[0]
        bars = [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]
        self.vis_objects = [TimeVolumeProfile(time=h["Date"][1], hist=bars)]