import numpy as np


class VolDistribution:
    def __init__(self):
        pass
    
    def update(self, h):
        x_bins = np.linspace(h["Low"][:-1].min(), h["High"][:-1].max(), 20)
        x = (h["High"][:-1] + h["Low"][:-1])/2
        y = h["Volume"][:-1]
        hist = np.histogram(x, bins=x_bins, weights=y)[0]
        return [(x, y) for x, y in zip(x_bins, hist)]