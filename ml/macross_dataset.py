"""Build direction-labeled dataset from OHLCV using MA cross strategy (y=1 when MA fast > MA slow, else y=-1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import PyConfig
from data_processing.dataloading import MovingWindow

DEFAULT_CONFIG = REPO_ROOT / "configs" / "macross" / "BTCUSDT.py"
FEATURE_WINDOW = 30  # last L closes for features
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


def _ensure_findata() -> None:
    if os.environ.get("FINDATA") is None:
        os.environ["FINDATA"] = str(REPO_ROOT / "fin_data")


def build_direction_dataset(config_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load BTCUSDT (or config-specified symbol) OHLCV, compute MA-cross direction labels,
    and build per-row features. Default config: configs/macross/BTCUSDT.py (overridable via MACROSS_RF_CONFIG).

    Returns:
        X: (n_samples, n_features) float array
        y: (n_samples,) int array with values in {-1, 1}
        meta: dict with timestamps, open_price, aligned_rows, class_balance, feature_names
    """
    _ensure_findata()
    config_path = config_path or os.environ.get("MACROSS_RF_CONFIG", str(DEFAULT_CONFIG))
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()

    pc = PyConfig(str(config_path))
    cfg = pc.get_backtest()
    raw_backtest = pc.base_config.backtest
    hist_size = int(cfg["hist_size"])
    dm = raw_backtest.get("decision_maker") or {}
    ma_fast_period_param = int(dm.get("ma_fast_period", 16))
    ma_slow_period = hist_size
    ma_fast_period = max(1, hist_size // ma_fast_period_param)

    mw = MovingWindow(cfg)
    raw_count = len(mw)
    n = raw_count
    if n == 0:
        X = np.zeros((0, 2), dtype=np.float64)
        y = np.array([], dtype=np.int64)
        timestamps = np.array([], dtype=mw.hist["Date"].dtype)
        open_price = np.array([], dtype=np.float64)
    else:
        X = np.zeros((n, 2), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)
        timestamps = np.empty(n, dtype=mw.hist["Date"].dtype)
        open_price = np.empty(n, dtype=np.float64)

        k = 0
        for window in mw(output_time=False):
            close_window = window["Close"]
            ma_fast = float(close_window[-ma_fast_period:].mean())
            ma_slow = float(close_window[-ma_slow_period:].mean())
            X[k, 0] = ma_slow
            X[k, 1] = ma_fast
            y[k] = 1 if ma_fast > ma_slow else -1
            timestamps[k] = window["Date"][-1]
            open_price[k] = float(window["Open"][-1])
            k += 1

        if k != n:
            X = X[:k]
            y = y[:k]
            timestamps = timestamps[:k]
            open_price = open_price[:k]

    unique, counts = np.unique(y, return_counts=True)
    class_balance = dict(zip(unique.tolist(), counts.tolist()))
    feature_names = ["ma_slow", "ma_fast"]
    meta = {
        "timestamps": timestamps,
        "open_price": open_price,
        "aligned_rows": int(X.shape[0]),
        "class_balance": class_balance,
        "feature_names": feature_names,
    }
    return X, y, meta


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, np.ndarray]:
    """
    Split (X, y) by time: first train_frac for train, next val_frac for val, rest for test.
    Assumes rows are already in chronological order.
    """
    n = X.shape[0]
    if n == 0:
        return {
            "X_train": np.zeros((0, X.shape[1]), dtype=X.dtype),
            "y_train": np.array([], dtype=y.dtype),
            "X_val": np.zeros((0, X.shape[1]), dtype=X.dtype),
            "y_val": np.array([], dtype=y.dtype),
            "X_test": np.zeros((0, X.shape[1]), dtype=X.dtype),
            "y_test": np.array([], dtype=y.dtype),
        }
    train_end = int(np.round(n * train_frac))
    val_end = int(np.round(n * (train_frac + val_frac)))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
    }
