"""Build MA-based feature dataset from OHLCV with buy-and-hold profit metadata."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import PyConfig
from data_processing.dataloading import MovingWindow

TRAIN_FRAC = 0.8


@dataclass
class DatasetMeta:
    """Metadata returned with `X` from `build_direction_dataset`."""

    timestamps: np.ndarray
    open_price: np.ndarray
    aligned_rows: int
    feature_names: list[str]
    buy_and_hold_step_profit: np.ndarray
    buy_and_hold_cum_profit: np.ndarray
    buy_and_hold_final_profit: float


def build_direction_dataset(config_path: str | Path | None = None) -> tuple[np.ndarray, DatasetMeta]:
    """
    Load BTCUSDT (or config-specified symbol) OHLCV and build per-row MA features.
    Default config: configs/macross/BTCUSDT.py (overridable via MACROSS_RF_CONFIG).

    Returns:
        X: (n_samples, n_features) float array
        meta: timestamps, open_price, aligned_rows, feature_names,
              buy-and-hold profit series and final value
    """
    if os.environ.get("FINDATA") is None:
            os.environ["FINDATA"] = str(REPO_ROOT / "fin_data")
    cfg = PyConfig(str(config_path)).base_config.config
    hist_size = int(cfg["hist_size"])
    ma_divisors = {
        "ma_10_period": 10,
        "ma_20_period": 20,
        }

    ma_feature_names = [name[:-7] for name in ma_divisors]
    ma_periods = {
        name: max(1, hist_size // max(1, int(divisor)))
        for name, divisor in ma_divisors.items()
    }
    # Per period: mean(close)/last_close, std(close)/last_close, std(volume)/last_volume
    n_features = len(ma_feature_names) * 3 + 2
    feature_names = [
        label
        for base in ma_feature_names
        for label in (base, f"{base}_std", f"{base}_vol_std")
    ]

    mw = MovingWindow(cfg)
    raw_count = len(mw)
    n = raw_count
    if n == 0:
        X = np.zeros((0, n_features), dtype=np.float64)
        timestamps = np.array([], dtype=mw.hist["Date"].dtype)
        open_price = np.array([], dtype=np.float64)
    else:
        X = np.zeros((n, n_features), dtype=np.float64)
        timestamps = np.empty(n, dtype=mw.hist["Date"].dtype)
        open_price = np.empty(n, dtype=np.float64)

        k = 0
        for window in mw(output_time=False):
            close_window = window["Close"][:-1]
            vol_window = window["Volume"][:-1]
            denom_price = close_window.mean()
            denom_vol = vol_window.mean()
            assert denom_price > 1e-15
            assert denom_vol > 1e-15
            col = 0
            for period_name in ma_divisors:
                p = ma_periods[period_name]
                close_slice = close_window[-p:]
                vol_slice = vol_window[-p:]

                X[k, col] = float(close_slice.mean()) / denom_price
                col += 1
                X[k, col] = float(np.std(close_slice, ddof=0)) / denom_price
                col += 1
                X[k, col] = float(np.std(vol_slice, ddof=0)) / denom_vol
                col += 1

            X[k, col] = float(vol_slice[-1]) / denom_vol
            col += 1
            X[k, col] = float(close_slice[-1]) / denom_price
            col += 1
            timestamps[k] = window["Date"][-1]
            open_price[k] = float(window["Open"][-1])
            k += 1

        if k != n:
            X = X[:k]
            timestamps = timestamps[:k]
            open_price = open_price[:k]

    buy_and_hold_step_profit = np.diff(open_price) if open_price.size > 1 else np.array([], dtype=np.float64)
    buy_and_hold_cum_profit = np.cumsum(buy_and_hold_step_profit) if buy_and_hold_step_profit.size else np.array([], dtype=np.float64)
    buy_and_hold_final_profit = float(buy_and_hold_cum_profit[-1]) if buy_and_hold_cum_profit.size else 0.0
    meta = DatasetMeta(
        timestamps=timestamps,
        open_price=open_price,
        aligned_rows=int(X.shape[0]),
        feature_names=feature_names,
        buy_and_hold_step_profit=buy_and_hold_step_profit,
        buy_and_hold_cum_profit=buy_and_hold_cum_profit,
        buy_and_hold_final_profit=buy_and_hold_final_profit,
    )
    return X, meta

