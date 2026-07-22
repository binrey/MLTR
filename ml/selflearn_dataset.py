"""Build compact dimensionless feature dataset from OHLCV."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import PyConfig
from data_processing.dataloading import MovingWindow

TRAIN_FRAC = 0.8
EPS = 1e-12
TRADING_DAYS = 252.0
FEATURE_CLIP = 5.0
CACHE_VERSION = 1
CACHE_ROOT = REPO_ROOT / ".cache" / "selflearn_dataset"

RETURN_PERIODS = [1, 4, 8, 16, 32, 64, 128]
MOMENTUM_Z_PERIODS = [8, 16, 32, 64, 128]
MA_PERIODS = [16, 32, 64, 128]
EFFICIENCY_PERIODS = [8, 16, 32, 64]
VOL_PERIODS = [8, 16, 32, 64, 128]
VOL_ASYMMETRY_PERIODS = [32, 64]
RANGE_PERIODS = [16, 32, 64, 128, 200]
DRAWDOWN_PERIODS = [32, 64, 128, 200]
TIME_SINCE_HIGH_PERIODS = [64, 128]
UP_RATIO_PERIODS = [16, 32, 64]
ACF_PERIODS = [16, 32, 64]
SKEW_PERIODS = [32, 64]
VOLUME_Z_PERIODS = [16, 32, 64]
SIGNED_VOLUME_PERIODS = [16, 32]
PRICE_VOLUME_CORR_PERIODS = [32, 64]


def _feature_spec_hash() -> str:
    payload = json.dumps(_compact_feature_names(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _resolve_data_file(cfg: PyConfig) -> Path:
    database = Path(os.environ.get("FINDATA", "../fin_data"))
    p = database / cfg["data_type"] / cfg["period"].value
    flist = [f for f in p.glob("*") if cfg["symbol"].ticker in f.stem]
    if not flist:
        raise FileNotFoundError(f"No data for {cfg['symbol'].ticker} in {p}")
    return flist[np.argmin([len(f.name) for f in flist])]


def _dataset_cache_params(cfg: PyConfig) -> dict:
    source = _resolve_data_file(cfg)
    return {
        "cache_version": CACHE_VERSION,
        "hist_size": int(cfg["hist_size"]),
        "date_start": str(np.datetime64(cfg["date_start"])),
        "date_end": str(np.datetime64(cfg["date_end"])),
        "period": cfg["period"].value,
        "data_type": cfg["data_type"],
        "symbol": cfg["symbol"].ticker,
        "feature_spec": _feature_spec_hash(),
        "source_path": str(source.resolve()),
        "source_mtime": float(source.stat().st_mtime),
    }


def _dataset_cache_dir(params: dict) -> Path:
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:32]
    return CACHE_ROOT / digest


def _try_load_single_from_cache(
    cfg: PyConfig, feature_names: list[str]
) -> tuple[np.ndarray, DatasetMeta] | None:
    params = _dataset_cache_params(cfg)
    cache_dir = _dataset_cache_dir(params)
    csv_path = cache_dir / "dataset.csv"
    params_path = cache_dir / "params.json"
    if not csv_path.exists() or not params_path.exists():
        return None

    stored_params = json.loads(params_path.read_text())
    if stored_params != params:
        logger.info(
            f"Selflearn dataset cache stale for {cfg['symbol'].ticker}, rebuilding"
        )
        return None

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    expected_cols = {"timestamp", "open_price", *feature_names}
    if set(df.columns) != expected_cols:
        logger.info(
            f"Selflearn dataset cache columns mismatch for {cfg['symbol'].ticker}, rebuilding"
        )
        return None

    n_features = len(feature_names)
    X = df[feature_names].to_numpy(dtype=np.float64)
    if X.shape[1] != n_features:
        return None

    timestamps = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ms]")
    open_price = df["open_price"].to_numpy(dtype=np.float64)
    meta = DatasetMeta(
        timestamps=timestamps,
        open_price=open_price,
        aligned_rows=int(X.shape[0]),
        feature_names=feature_names,
        symbols=[cfg["symbol"]],
    )
    logger.info(
        f"Loaded selflearn dataset cache for {cfg['symbol'].ticker} "
        f"({X.shape[0]} rows) from {csv_path}"
    )
    return X, meta


def _save_single_to_cache(
    cfg: PyConfig,
    X: np.ndarray,
    meta: DatasetMeta,
    feature_names: list[str],
) -> None:
    params = _dataset_cache_params(cfg)
    cache_dir = _dataset_cache_dir(params)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "dataset.csv"
    params_path = cache_dir / "params.json"

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(meta.timestamps),
            "open_price": meta.open_price,
            **{name: X[:, i] for i, name in enumerate(feature_names)},
        }
    )
    frame.to_csv(csv_path, index=False)
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True))
    logger.info(
        f"Saved selflearn dataset cache for {cfg['symbol'].ticker} "
        f"({X.shape[0]} rows) to {csv_path}"
    )


def _compact_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"return_{p}" for p in RETURN_PERIODS)
    names.extend(f"momentum_z_{p}" for p in MOMENTUM_Z_PERIODS)
    names.extend(f"price_ma_z_{p}" for p in MA_PERIODS)
    names.extend(["ma_16_32_z", "ma_32_128_z", "ma_16_128_z"])
    names.extend(f"signed_efficiency_{p}" for p in EFFICIENCY_PERIODS)
    names.extend(f"volatility_{p}" for p in VOL_PERIODS)
    names.extend(["vol_ratio_16_64", "vol_ratio_32_128"])
    names.extend(f"vol_asymmetry_{p}" for p in VOL_ASYMMETRY_PERIODS)
    names.append("return_shock_32")
    names.extend(f"range_pos_{p}" for p in RANGE_PERIODS)
    names.extend(f"drawdown_{p}" for p in DRAWDOWN_PERIODS)
    names.extend(f"time_since_high_{p}" for p in TIME_SINCE_HIGH_PERIODS)
    names.extend(f"up_ratio_{p}" for p in UP_RATIO_PERIODS)
    names.extend(f"return_acf_lag1_{p}" for p in ACF_PERIODS)
    names.extend(f"return_skew_{p}" for p in SKEW_PERIODS)
    names.extend(f"volume_z_{p}" for p in VOLUME_Z_PERIODS)
    names.append("volume_trend_16_64")
    names.extend(f"signed_volume_{p}" for p in SIGNED_VOLUME_PERIODS)
    names.extend(f"price_volume_corr_{p}" for p in PRICE_VOLUME_CORR_PERIODS)
    names.extend(
        [
            "intraday_return",
            "overnight_gap",
            "daily_range",
            "candle_body",
            "upper_shadow",
            "lower_shadow",
        ]
    )
    return names


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log(max(float(numerator), EPS) / max(float(denominator), EPS)))


def _return_over(close: np.ndarray, period: int) -> float:
    return _safe_log_ratio(float(close[-1]), float(close[-period - 1]))


def _std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values, ddof=0))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = _std(x)
    y_std = _std(y)
    if x_std <= EPS or y_std <= EPS:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def _skew(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    centered = values - np.mean(values)
    std = _std(centered)
    if std <= EPS:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _clip_feature(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -FEATURE_CLIP, FEATURE_CLIP))


def _compact_features(
    open_hist: np.ndarray,
    high_hist: np.ndarray,
    low_hist: np.ndarray,
    close_hist: np.ndarray,
    volume_hist: np.ndarray,
) -> list[float]:
    close = np.asarray(close_hist, dtype=np.float64)
    open_ = np.asarray(open_hist, dtype=np.float64)
    high = np.asarray(high_hist, dtype=np.float64)
    low = np.asarray(low_hist, dtype=np.float64)
    volume = np.asarray(volume_hist, dtype=np.float64)

    price = float(close[-1])
    log_returns = np.diff(np.log(np.maximum(close, EPS)))
    log_volume = np.log1p(np.maximum(volume, 0.0))
    volume_change = np.diff(log_volume)
    vol_by_period = {p: _std(log_returns[-p:]) for p in VOL_PERIODS}
    price_std_by_period = {p: _std(close[-p:]) for p in MA_PERIODS}

    features: list[float] = []

    for p in RETURN_PERIODS:
        features.append(_return_over(close, p))

    for p in MOMENTUM_Z_PERIODS:
        ret = _return_over(close, p)
        denom = vol_by_period[p] * np.sqrt(float(p)) + EPS
        features.append(ret / denom)

    ma_values = {p: float(np.mean(close[-p:])) for p in MA_PERIODS}
    for p in MA_PERIODS:
        features.append((price - ma_values[p]) / (price_std_by_period[p] + EPS))

    features.append((ma_values[16] - ma_values[32]) / (price_std_by_period[32] + EPS))
    features.append((ma_values[32] - ma_values[128]) / (price_std_by_period[128] + EPS))
    features.append((ma_values[16] - ma_values[128]) / (price_std_by_period[128] + EPS))

    for p in EFFICIENCY_PERIODS:
        path = float(np.sum(np.abs(np.diff(close[-(p + 1):]))))
        features.append((price - float(close[-p - 1])) / (path + EPS))

    for p in VOL_PERIODS:
        features.append(vol_by_period[p] * np.sqrt(TRADING_DAYS))

    features.append(vol_by_period[16] / (vol_by_period[64] + EPS))
    features.append(vol_by_period[32] / (vol_by_period[128] + EPS))

    for p in VOL_ASYMMETRY_PERIODS:
        ret_slice = log_returns[-p:]
        downside = ret_slice[ret_slice < 0.0]
        upside = ret_slice[ret_slice > 0.0]
        features.append(_std(downside) / (_std(upside) + EPS))

    features.append(float(log_returns[-1]) / (vol_by_period[32] + EPS))

    for p in RANGE_PERIODS:
        price_slice = close[-p:]
        low_p = float(np.min(price_slice))
        high_p = float(np.max(price_slice))
        features.append((price - low_p) / (high_p - low_p + EPS))

    for p in DRAWDOWN_PERIODS:
        features.append(price / (float(np.max(close[-p:])) + EPS) - 1.0)

    for p in TIME_SINCE_HIGH_PERIODS:
        price_slice = close[-p:]
        high_idx = int(np.argmax(price_slice))
        features.append(float(p - 1 - high_idx) / float(p))

    for p in UP_RATIO_PERIODS:
        features.append(float(np.mean(log_returns[-p:] > 0.0)))

    for p in ACF_PERIODS:
        ret_slice = log_returns[-(p + 1):]
        features.append(_corr(ret_slice[1:], ret_slice[:-1]))

    for p in SKEW_PERIODS:
        features.append(_skew(log_returns[-p:]))

    for p in VOLUME_Z_PERIODS:
        log_vol_slice = log_volume[-p:]
        features.append((float(log_volume[-1]) - float(np.mean(log_vol_slice))) / (_std(log_vol_slice) + EPS))

    features.append(float(np.mean(volume[-16:])) / (float(np.mean(volume[-64:])) + EPS) - 1.0)

    for p in SIGNED_VOLUME_PERIODS:
        ret_slice = log_returns[-p:]
        vol_slice = volume[-p:]
        features.append(float(np.sum(np.sign(ret_slice) * vol_slice)) / (float(np.sum(vol_slice)) + EPS))

    for p in PRICE_VOLUME_CORR_PERIODS:
        features.append(_corr(log_returns[-p:], volume_change[-p:]))

    last_range = max(float(high[-1] - low[-1]), EPS)
    prev_close = float(close[-2])
    features.extend(
        [
            _safe_log_ratio(float(close[-1]), float(open_[-1])),
            _safe_log_ratio(float(open_[-1]), prev_close),
            (float(high[-1] - low[-1])) / (prev_close + EPS),
            (float(close[-1] - open_[-1])) / last_range,
            (float(high[-1] - max(open_[-1], close[-1]))) / last_range,
            (float(min(open_[-1], close[-1]) - low[-1])) / last_range,
        ]
    )

    return [_clip_feature(value) for value in features]


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
    Load OHLCV and build per-row compact dimensionless features.

    Returns:
        X: (n_samples, n_features) float array
        meta: timestamps, open_price, aligned_rows, feature_names
    """
    hist_size = int(cfg["hist_size"])
    required_history = max(RANGE_PERIODS + [max(RETURN_PERIODS) + 1])
    if hist_size < required_history:
        raise ValueError(
            f"hist_size={hist_size} is too small for compact features; "
            f"expected at least {required_history}"
        )

    feature_names = _compact_feature_names()
    n_features = len(feature_names)

    cached = _try_load_single_from_cache(cfg, feature_names)
    if cached is not None:
        return cached

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
            features = _compact_features(
                open_hist=window["Open"][:-1],
                high_hist=window["High"][:-1],
                low_hist=window["Low"][:-1],
                close_hist=window["Close"][:-1],
                volume_hist=window["Volume"][:-1],
            )
            if len(features) != n_features:
                raise RuntimeError(
                    f"Feature count mismatch: got {len(features)}, expected {n_features}"
                )
            X[k, :] = features

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
    _save_single_to_cache(cfg, X, meta, feature_names)
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
    Load BTCUSDT (or config-specified symbol) OHLCV and build per-row features.
    Default config: configs/macross/BTCUSDT.py (overridable via MACROSS_RF_CONFIG).

    Returns:
        X: (n_samples, n_features) float array
        meta: timestamps, open_price, aligned_rows, feature_names
    """
    cfg = PyConfig(str(config_path)).base_config.config
    if cfg["symbols"] is None:
        raise ValueError("symbols is not set")
    return build_multi_simbol_dataset(cfg)