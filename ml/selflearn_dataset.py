"""Build MA-based feature dataset from OHLCV."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
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
    """Metadata returned with `X` from dataset builders.

    Multi-symbol builds may use NaN in ``open_price`` and in ``X`` where a
    symbol has no bar on a date present for another symbol.
    """

    timestamps: np.ndarray
    open_price: np.ndarray
    aligned_rows: int
    feature_names: list[str]
    symbols: list[str]


def build_single_simbol_dataset(cfg: PyConfig) -> tuple[np.ndarray, DatasetMeta]:
    """
    Load OHLCV and build per-row MA features.

    Returns:
        X: (n_samples, n_features) float array
        meta: timestamps, open_price, aligned_rows, feature_names
    """
    hist_size = int(cfg["hist_size"])
    ma_divisors = {
        "ma_10_period": 10,
        "ma_40_period": 40,
        }

    ma_feature_names = [name[:-7] for name in ma_divisors]
    ma_periods = {
        name: max(1, hist_size // max(1, int(divisor)))
        for name, divisor in ma_divisors.items()
    }
    # Per period: mean(close)/last_close, std(close)/last_close, std(volume)/last_volume
    # Plus current normalized values and cumulative drawdown stats.
    n_features = len(ma_feature_names) * 3 + 2
    feature_names = [
        label
        for base in ma_feature_names
        for label in (base, f"{base}_std", f"{base}_vol_std")
    ]
    feature_names.extend(
        [
            "last_vol_rel",
            "last_close_rel",
            # "drawdown_price",
            # "drawdown_periods",
        ]
    )

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
        running_peak_close = -np.inf
        drawdown_periods = 0
        drawdown_initialized = False
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

                # mean close price
                X[k, col] = float(close_slice.mean()) / denom_price
                # std close price
                col += 1
                X[k, col] = float(np.std(close_slice, ddof=0)) / denom_price
                # std volume
                col += 1
                X[k, col] = float(np.std(vol_slice, ddof=0)) / denom_vol
                col += 1

            # last volume relative to mean volume
            X[k, col] = float(vol_slice[-1]) / denom_vol

            # last close price
            col += 1
            X[k, col] = float(close_slice[-1]) / denom_price

            # # running peak close price
            # col += 1
            # current_close = float(close_window[-1])
            # if not drawdown_initialized:
            #     peak_index = int(np.argmax(close_window))
            #     running_peak_close = float(close_window[peak_index])
            #     drawdown_periods = int(close_window.shape[0] - 1 - peak_index)
            #     drawdown_initialized = True
            # elif current_close >= running_peak_close:
            #     running_peak_close = current_close
            #     drawdown_periods = 0
            # else:
            #     drawdown_periods += 1

            # X[k, col] = max(0.0, running_peak_close - current_close) / running_peak_close

            # # drawdown periods
            # col += 1
            # X[k, col] = float(drawdown_periods) / hist_size

            timestamps[k] = window["Date"][-1]
            open_price[k] = float(window["Open"][-1])

            k += 1

        if k != n:
            X = X[:k]
            timestamps = timestamps[:k]
            open_price = open_price[:k]

    meta = DatasetMeta(
        timestamps=timestamps,
        open_price=open_price,
        aligned_rows=int(X.shape[0]),
        feature_names=feature_names,
        symbols=[cfg["symbol"]],
    )
    return X, meta


def build_multi_simbol_dataset(cfg: PyConfig) -> tuple[np.ndarray, DatasetMeta]:
    """
    Load OHLCV per symbol and keep each symbol's native date sequence.

    Output is panel-shaped and padded to the longest symbol history:
    - ``X``: (n_symbols, n_times_max, n_features)
    - ``timestamps``: (n_symbols, n_times_max)
    - ``open_price``: (n_symbols, n_times_max)

    Padding is appended at the tail (NaN/NaT), so every symbol starts with
    non-missing rows at index 0.

    Returns:
        X: (n_batch, n_times, n_features) float array (may contain NaN)
        meta: timestamps, open_price, aligned_rows, feature_names
    """
    symbols = cfg["symbols"]
    assert isinstance(symbols, list), f"build_multi_simbol_dataset expects cfg['symbols'] to be a list, got {type(symbols)}"

    per_X: list[np.ndarray] = []
    per_ts: list[np.ndarray] = []
    per_open: list[np.ndarray] = []
    base_feature_names: list[str] | None = None

    for sym in symbols:
        sub = deepcopy(cfg)
        sub["symbol"] = sym
        X_i, meta_i = build_single_simbol_dataset(sub)
        if base_feature_names is None:
            base_feature_names = list(meta_i.feature_names)
        elif meta_i.feature_names != base_feature_names:
            raise ValueError(
                f"Feature names mismatch for {sym.ticker} vs {symbols[0].ticker}"
            )
        per_X.append(X_i)
        per_ts.append(np.asarray(meta_i.timestamps))
        per_open.append(np.asarray(meta_i.open_price, dtype=np.float64))

    assert base_feature_names is not None
    n_features = len(base_feature_names)
    n_times_max = max((int(ts.shape[0]) for ts in per_ts), default=0)
    if n_times_max == 0:
        X = np.zeros((len(symbols), 0, n_features), dtype=np.float64)
        timestamps = np.zeros((len(symbols), 0), dtype=np.dtype("datetime64[ms]"))
        open_price = np.zeros((len(symbols), 0), dtype=np.float64)
    else:
        ts_dtype = per_ts[0].dtype
        X = np.full((len(symbols), n_times_max, n_features), np.nan, dtype=np.float64)
        timestamps = np.full((len(symbols), n_times_max), np.datetime64("NaT"), dtype=ts_dtype)
        open_price = np.full((len(symbols), n_times_max), np.nan, dtype=np.float64)
        for i, (X_i, ts_i, op_i) in enumerate(zip(per_X, per_ts, per_open)):
            n_i = int(ts_i.shape[0])
            if n_i == 0:
                continue
            X[i, -n_i:, :] = X_i
            timestamps[i, -n_i:] = ts_i
            open_price[i, -n_i:] = op_i

    meta = DatasetMeta(
        timestamps=timestamps,
        open_price=open_price,
        aligned_rows=int(X.shape[0] * X.shape[1]),
        feature_names=base_feature_names,
        symbols=symbols,
    )
    return X, meta


def build_dataset(config_path: str | Path | None = None) -> tuple[np.ndarray, DatasetMeta]:
    """
    Load BTCUSDT (or config-specified symbol) OHLCV and build per-row MA features.
    Default config: configs/macross/BTCUSDT.py (overridable via MACROSS_RF_CONFIG).

    Returns:
        X: (n_samples, n_features) float array
        meta: timestamps, open_price, aligned_rows, feature_names
    """
    cfg = PyConfig(str(config_path)).base_config.config
    if cfg["symbols"] is None:
        raise ValueError("symbols is not set")
    return build_multi_simbol_dataset(cfg)