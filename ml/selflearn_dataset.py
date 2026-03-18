"""Build MA-based feature dataset from OHLCV with buy-and-hold profit metadata."""

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
TRAIN_FRAC = 0.8


def build_direction_dataset(config_path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """
    Load BTCUSDT (or config-specified symbol) OHLCV and build per-row MA features.
    Default config: configs/macross/BTCUSDT.py (overridable via MACROSS_RF_CONFIG).

    Returns:
        X: (n_samples, n_features) float array
        meta: dict with timestamps, open_price, aligned_rows, feature_names,
              buy_and_hold_step_profit, buy_and_hold_cum_profit, buy_and_hold_final_profit
    """
    if os.environ.get("FINDATA") is None:
            os.environ["FINDATA"] = str(REPO_ROOT / "fin_data")
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
    # All MA periods are defined as fractions of hist_size: hist_size // divisor.
    ma_divisors = {
        "ma_slow_period": 1,
        "ma_10_period": 10,
        "ma_20_period": 20,
        "ma_40_period": 40,
        "ma_40_period": 80,
    }

    ma_feature_names = [name[:-7] for name in ma_divisors]
    ma_periods = {
        name: max(1, hist_size // max(1, int(divisor)))
        for name, divisor in ma_divisors.items()
    }

    mw = MovingWindow(cfg)
    raw_count = len(mw)
    n = raw_count
    n_features = len(ma_feature_names)
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
            close_window = window["Close"]
            for idx, period_name in enumerate(ma_divisors):
                X[k, idx] = float(close_window[-ma_periods[period_name] :].mean())
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
    feature_names = ma_feature_names
    meta = {
        "timestamps": timestamps,
        "open_price": open_price,
        "aligned_rows": int(X.shape[0]),
        "feature_names": feature_names,
        "buy_and_hold_step_profit": buy_and_hold_step_profit,
        "buy_and_hold_cum_profit": buy_and_hold_cum_profit,
        "buy_and_hold_final_profit": buy_and_hold_final_profit,
    }
    return X, meta

