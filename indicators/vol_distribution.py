"""Module for analyzing volume distribution across different price levels."""

import os
import pickle
from pathlib import Path

import numpy as np

from backtesting.profiling import line_profile_function, profile_function
from common.type import TimeVolumeProfile


class VolDistribution:
    """Calculates volume distribution across price levels using histogram binning."""
    def __init__(self, nbins=20):
        self.nbins = nbins
        self.vol_hist, self.price_bins = None, None
        self.vol_profile = None
        self.cache = {}
        self.cache_dir = Path(os.getenv("CACHE_DIR")) / "vol_distribution"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def __del__(self):
        self._save_cache()

    def _get_cache_path(self):
        return self.cache_dir / f"nbins={self.nbins}.pkl"

    def _load_cache(self):
        cache_file = self._get_cache_path()
        if cache_file.exists():
                with open(cache_file, "rb") as f:
                        self.cache = pickle.load(f)

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
        return f"{str(h['Date'][1])}_{self.nbins}"

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

        # If not in cache, calculate and cache the results
        self._update_without_cache(h)
        
        # Save to cache
        self._save_to_cache(cache_key)
        
        return self.vol_profile

    def _update_without_cache(self, h: np.ndarray) -> TimeVolumeProfile:
        """Update method without caching logic"""
        upper_price = h["High"][:-1].max()
        lower_price = h["Low"][:-1].min()
        self.price_bins = np.linspace(lower_price, upper_price, self.nbins)

        upper = h["High"][:-1] - np.vstack([h["Close"][:-1], h["Open"][:-1]]).max(0)
        lower = np.vstack([h["Close"][:-1], h["Open"][:-1]]).min(0) - h["Low"][:-1]        
        x = (h["High"][:-1]*upper + h["Low"][:-1]*lower)/(upper + lower + 0.001)
        y = h["Volume"][:-1]

        self.vol_hist = self.histogram(x, bins=self.price_bins, weights=y)
        bars = [(x, y) for x, y in zip(self.price_bins, self.vol_hist)]
        self.vol_profile = TimeVolumeProfile(time=h["Date"][1], hist=bars)
        return self.vol_profile
        
    @property
    def vis_objects(self):
        return [self.vol_profile]


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