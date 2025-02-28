"""Module for analyzing volume distribution across different price levels."""

import numpy as np

from common.type import TimeVolumeProfile


class VolDistribution:
    """Calculates volume distribution across price levels using histogram binning."""
    def __init__(self, nbins=20):
        self.nbins = nbins
        self.vol_hist, self.price_bins = None, None
        self.vol_profile = None
    
    def update(self, h) -> TimeVolumeProfile:
        self.price_bins = np.linspace(h["Low"][:-1].min(), h["High"][:-1].max(), self.nbins)
        # V1
        upper = h["High"][:-1] - np.vstack([h["Close"][:-1], h["Open"][:-1]]).max(0)
        lower = np.vstack([h["Close"][:-1], h["Open"][:-1]]).min(0) - h["Low"][:-1]        
        x = (h["High"][:-1]*upper + h["Low"][:-1]*lower)/(upper + lower + 0.001)
        y = h["Volume"][:-1]

        # V2
        # upper = h["High"][:-1] - np.vstack([h["Close"][:-1], h["Open"][:-1]]).max(0)
        # lower = np.vstack([h["Close"][:-1], h["Open"][:-1]]).min(0) - h["Low"][:-1]      
        # x = np.concatenate([h["High"][:-1], h["Low"][:-1]])
        # sum_size = lower + upper
        # k = upper / (sum_size + 0.00001)
        # y = np.concatenate([k * h["Volume"][:-1], (1-k) * h["Volume"][:-1]])

        self.vol_hist = np.histogram(x, bins=self.price_bins, weights=y)[0]
        bars = [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]
        self.vol_profile = TimeVolumeProfile(time=h["Date"][1], hist=bars)
        
    @property
    def vis_objects(self):
        return [self.vol_profile]