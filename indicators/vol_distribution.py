"""Module for analyzing volume distribution across different price levels."""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from common.type import TimeVolumeProfile


class VolDistribution:
    """Calculates volume distribution across price levels using histogram binning."""
    def __init__(self, cache_dir: Optional[str] = None):
        self.vol_hist, self.price_bins = None, None
        self.vol_profile = None
        self.cache = {}
        self.cache_dir = Path(cache_dir) / "vol_distribution" if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def __del__(self):
        self._save_cache()

    def _get_cache_path(self):
        return self.cache_dir / "cache.pkl"

    def _load_cache(self):
        cache_file = self._get_cache_path()
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, IOError):
                # cache_file.unlink()
                # self.cache = {}
                logger.warning(f"Corrupted cache file {cache_file}")

    def _save_cache(self):
        if not self.cache:
            return
        cache_file = self._get_cache_path()
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def histogram(data, bins, weights):
        # Find the bin index for each data point
        inds = np.searchsorted(bins, data, side='right') - 1
        # Exclude values that fall outside the bin range
        valid = (inds >= 0) & (inds < len(bins) - 1)
        # Use np.bincount with weights
        hist = np.bincount(inds[valid], weights=weights[valid], minlength=len(bins) - 1)
        return hist

    def _get_cache_key(self, h):
        """Generate a unique cache key based on input data and parameters"""
        # Use the date and nbins as the cache key
        return f"{str(h['Date'][-1])}"

    def _load_from_cache(self, cache_key):
        """Load histogram data from cache if available"""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            self.vol_hist = cached_data['vol_hist']
            self.price_bins = cached_data['price_bins']
            return True
        return False

    def _save_to_cache(self, cache_key):
        """Save histogram data to cache"""
        self.cache[cache_key] = {
            'vol_hist': self.vol_hist,
            'price_bins': self.price_bins
        }
        # Periodically save to disk (every 10 updates)
        # if len(self.cache) % 1000 == 0:
        #     self._save_cache()

    def update(self, h: np.ndarray) -> TimeVolumeProfile:
        cache_key = self._get_cache_key(h)

        # Try to load from cache first
        if self._load_from_cache(cache_key):
            bars = [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]
            self.vol_profile = TimeVolumeProfile(time=h["Date"][1], hist=bars)
            return self.vol_profile

        self._update_without_cache(h)
        self._save_to_cache(cache_key)
        logger.debug("volume profile:\n" +
                        " ".join([f"{p:10.1f}" for (p, v) in self.vol_profile.hist]) + "\n" + 
                        " ".join([f"{v:10.1f}" for (p, v) in self.vol_profile.hist]))
        return self.vol_profile

    def _update_without_cache(self, h: np.ndarray) -> TimeVolumeProfile:
        """Update method without caching logic"""
        upper_price = h["High"][:-1].max()
        lower_price = h["Low"][:-1].min()
        max_open_cls = np.vstack([h["Close"][:-1], h["Open"][:-1]]).max(0)
        min_open_cls = np.vstack([h["Close"][:-1], h["Open"][:-1]]).min(0)
        mean_body_size = (max_open_cls - min_open_cls).mean()
        nbins = 2
        if mean_body_size > 0:
            nbins = int(np.ceil((upper_price - lower_price)/mean_body_size))
        self.price_bins = np.linspace(lower_price, upper_price, nbins)

        upper = h["High"][:-1] - max_open_cls
        lower = min_open_cls - h["Low"][:-1]
        x = (h["High"][:-1]*upper + h["Low"][:-1]*lower)/(upper + lower + 0.001)
        y = h["Volume"][:-1]

        self.vol_hist = self.histogram(x, bins=self.price_bins, weights=y)
        bars = [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]
        self.vol_profile = TimeVolumeProfile(time=h["Date"][1], hist=bars)
        assert len(self.vol_hist) > 0
        return self.vol_profile
        
    @property
    def vis_objects(self):
        return [self.vol_profile]
    
    @property
    def bin_size(self):
        return abs(self.price_bins[0] - self.price_bins[1]) if len(self.price_bins) else 0


def test_vol_distribution():
    """Test the VolDistribution class"""
    indc = VolDistribution()
    
    # Generate sample OHLCV data
    dates = np.arange('2024-01-01', '2024-01-11', dtype='datetime64[D]')
    n_samples = len(dates)
    
    # Create sample price data with some realistic patterns
    base_price = 100
    price_trend = np.linspace(0, 10, n_samples)  # Upward trend
    noise = np.random.normal(0, 1, n_samples)
    
    # Generate OHLCV data
    data = {
        'Date': dates,
        'Open': base_price + price_trend + noise,
        'High': base_price + price_trend + noise + np.random.uniform(0.5, 2, n_samples),
        'Low': base_price + price_trend + noise - np.random.uniform(0.5, 2, n_samples),
        'Close': base_price + price_trend + np.random.normal(0, 1, n_samples),
        'Volume': np.random.uniform(1000, 5000, n_samples)
    }
    
    # Convert to structured array
    dtype = [('Date', 'datetime64[D]'), ('Open', 'f8'), ('High', 'f8'), 
            ('Low', 'f8'), ('Close', 'f8'), ('Volume', 'f8')]
    h = np.array(list(zip(data['Date'], data['Open'], data['High'], 
                         data['Low'], data['Close'], data['Volume'])), dtype=dtype)
    
    # Update and test the indicator
    for _ in range(1):
        vol_profile = indc.update(h)

if __name__ == "__main__":
    test_vol_distribution()