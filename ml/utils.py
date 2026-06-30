from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from common.utils import PyConfig

BatchPeriod = Literal["year", "month"]


@dataclass(frozen=True)
class PackedYearlyTrainingData:
    X: np.ndarray
    open_price: np.ndarray
    deposit_multp: np.ndarray
    loss_mask: np.ndarray
    metadata: dict


def resolve_commission_rate_pct(config_path: str | Path | None = None) -> float:
    """Execution fee as percent of price per deal (matches FeeRate.order_execution_rate)."""
    env_val = os.environ.get("COMMISSION_PCT", "").strip()
    if env_val:
        return float(env_val)
    if config_path:
        cfg = PyConfig(str(config_path)).base_config.config
        fee_rate = cfg.get("fee_rate")
        if fee_rate is not None:
            return float(fee_rate.order_execution_rate)
    return 0.0


def calendar_years_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Per-row calendar year as int64 (numpy/pandas datetime64 bars)."""
    return calendar_period_keys_from_timestamps(timestamps, period="year")


def calendar_months_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Per-row calendar month key as int64 (year * 12 + month - 1)."""
    return calendar_period_keys_from_timestamps(timestamps, period="month")


def calendar_period_keys_from_timestamps(
    timestamps: np.ndarray,
    period: BatchPeriod = "year",
) -> np.ndarray:
    """Per-row calendar period key (year int or month-encoded int)."""
    dt_index = pd.DatetimeIndex(np.asarray(timestamps))
    if period == "year":
        return dt_index.year.to_numpy(dtype=np.int64)
    if period == "month":
        return (
            dt_index.year.astype(np.int64) * 12
            + (dt_index.month - 1).astype(np.int64)
        )
    raise ValueError(f"Unsupported batch period: {period!r}")


def _previous_period_key(period_key: int, period: BatchPeriod) -> int:
    return int(period_key) - 1


def _batch_period_metadata(period: BatchPeriod) -> tuple[str, str]:
    if period == "year":
        return (
            "calendar_year_with_previous_year_warmup",
            "mean_per_year_sequence",
        )
    return (
        "calendar_month_with_previous_month_warmup",
        "mean_per_month_sequence",
    )


def cv_fold_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv_abs_mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    mean_value = float(np.mean(arr))
    std_value = float(np.std(arr))
    return {
        "mean": mean_value,
        "std": std_value,
        "cv_abs_mean": float(std_value / max(abs(mean_value), 1e-6)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_valid_panel_mask(X: np.ndarray, open_price: np.ndarray) -> np.ndarray:
    """Build a validity mask for a panel of data."""
    return np.isfinite(open_price) & np.all(np.isfinite(X), axis=2)


def valid_timestamp_mask(timestamps: np.ndarray) -> np.ndarray:
    if np.issubdtype(timestamps.dtype, np.datetime64):
        return ~np.isnat(timestamps)
    return np.ones(timestamps.shape, dtype=bool)


def pack_yearly_training_batch(
    X_train: np.ndarray,
    timestamps_train: np.ndarray,
    open_price_train: np.ndarray,
    deposit_multp_train: np.ndarray,
    period: BatchPeriod = "month",
) -> PackedYearlyTrainingData:
    """Pack (previous period warm-up, current period target) sequences into one batch."""
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (B,T,F), got {X_train.shape}")
    if period not in {"year", "month"}:
        raise ValueError(f"Unsupported batch period: {period!r}")

    bsz, _, n_features = X_train.shape
    valid_panel = build_valid_panel_mask(X_train, open_price_train) & valid_timestamp_mask(
        timestamps_train
    )
    sequences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    sequence_meta: list[dict] = []
    skipped_first_periods: list[dict] = []
    target_periods: set[int] = set()
    warmup_periods: set[int] = set()
    period_label = "year" if period == "year" else "month"

    for symbol_idx in range(bsz):
        valid_idx = np.flatnonzero(valid_panel[symbol_idx])
        if valid_idx.size == 0:
            continue

        valid_ts = timestamps_train[symbol_idx, valid_idx]
        valid_period_keys = calendar_period_keys_from_timestamps(valid_ts, period=period)
        unique_periods = sorted(int(key) for key in np.unique(valid_period_keys))
        if unique_periods:
            skipped_first_periods.append(
                {
                    "symbol_index": symbol_idx,
                    period_label: int(unique_periods[0]),
                }
            )

        period_to_idx = {
            period_key: valid_idx[valid_period_keys == period_key]
            for period_key in unique_periods
        }
        for target_period in unique_periods[1:]:
            warmup_period = _previous_period_key(target_period, period)
            if warmup_period not in period_to_idx:
                continue
            warmup_idx = period_to_idx[warmup_period]
            target_idx = period_to_idx[target_period]
            if warmup_idx.size == 0 or target_idx.size < 2:
                continue

            seq_idx = np.concatenate([warmup_idx, target_idx])
            seq_loss_mask = np.zeros(seq_idx.shape[0], dtype=bool)
            seq_loss_mask[warmup_idx.shape[0]:] = True
            sequences.append(
                (
                    X_train[symbol_idx, seq_idx, :],
                    open_price_train[symbol_idx, seq_idx],
                    deposit_multp_train[symbol_idx, seq_idx],
                    seq_loss_mask,
                )
            )
            warmup_periods.add(int(warmup_period))
            target_periods.add(int(target_period))
            sequence_meta.append(
                {
                    "symbol_index": symbol_idx,
                    f"warmup_{period_label}": int(warmup_period),
                    f"target_{period_label}": int(target_period),
                    "warmup_rows": int(warmup_idx.size),
                    "target_rows": int(target_idx.size),
                }
            )

    if not sequences:
        raise ValueError(
            f"No {period_label}ly training sequences with a previous calendar "
            f"{period_label} warm-up and at least two target rows"
        )

    max_len = max(seq[0].shape[0] for seq in sequences)
    packed_bsz = len(sequences)
    X_packed = np.full((packed_bsz, max_len, n_features), np.nan, dtype=np.float64)
    open_packed = np.full((packed_bsz, max_len), np.nan, dtype=np.float64)
    deposit_packed = np.zeros((packed_bsz, max_len), dtype=np.float64)
    loss_mask = np.zeros((packed_bsz, max_len), dtype=bool)

    for i, (X_seq, open_seq, deposit_seq, seq_loss_mask) in enumerate(sequences):
        n_seq = X_seq.shape[0]
        X_packed[i, :n_seq, :] = X_seq
        open_packed[i, :n_seq] = open_seq
        deposit_packed[i, :n_seq] = deposit_seq
        loss_mask[i, :n_seq] = seq_loss_mask

    train_batching, loss_reduction = _batch_period_metadata(period)
    metadata = {
        "batch_period": period,
        "train_batching": train_batching,
        "loss_reduction": loss_reduction,
        "original_batch_size": int(bsz),
        "original_timesteps": int(X_train.shape[1]),
        "packed_batch_size": int(packed_bsz),
        "packed_timesteps": int(max_len),
        f"target_{period_label}s": sorted(target_periods),
        f"warmup_{period_label}s": sorted(warmup_periods),
        f"skipped_first_{period_label}s": skipped_first_periods,
        "sequences": sequence_meta,
    }
    return PackedYearlyTrainingData(
        X=X_packed,
        open_price=open_packed,
        deposit_multp=deposit_packed,
        loss_mask=loss_mask,
        metadata=metadata,
    )


def compute_step_profit_with_boundaries(
    timestamps: np.ndarray,
    open_price: np.ndarray,
    direction: np.ndarray,
    deposit_multp: np.ndarray,
    valid_rows: np.ndarray,
    commission_rate_pct: float = 0.0,
) -> np.ndarray:
    """
    Per-step pnl with boundary handling.

    Returns length n-1 vector; invalid transitions or sequence boundaries are 0.
    Commissions: price * |Δposition| * commission_rate_pct / 100 on each deal.
    """
    n = int(open_price.shape[1])
    assert n > 1
    step_profit = np.zeros((open_price.shape[0], n - 1), dtype=np.float64)

    assert np.any(valid_rows)
    open_change = np.diff(open_price, axis=-1)
    n_steps = n - 1
    for i in range(open_price.shape[0]):
        vi = valid_rows[i]
        ts_i = timestamps[i]
        step_ok = vi[:-1] & vi[1:] & (ts_i[1:] > ts_i[:-1])
        step_profit[i, step_ok] = (
            open_change[i, step_ok]
            * direction[i, :n_steps][step_ok]
            * deposit_multp[i, :n_steps][step_ok]
        )
        if commission_rate_pct != 0.0:
            deal_volume = (
                np.abs(np.diff(direction[i, :n], axis=-1)[step_ok])
                * deposit_multp[i, :n_steps][step_ok]
            )
            step_profit[i, step_ok] -= (
                open_price[i, :n_steps][step_ok]
                * deal_volume
                * commission_rate_pct
                / 100.0
            )
    return step_profit


def compute_buy_hold_step_profit(
    timestamps: np.ndarray,
    open_price: np.ndarray,
    deposit: float,
) -> np.ndarray:
    """Always-long PnL with fixed deposit re-sized at every step."""
    n = int(open_price.shape[0])
    if n <= 1:
        return np.array([], dtype=np.float64)

    step_profit = np.zeros(n - 1, dtype=np.float64)
    valid_rows = np.isfinite(open_price) & (open_price > 0.0)
    step_ok = valid_rows[:-1] & valid_rows[1:] & (timestamps[1:] > timestamps[:-1])
    step_units = deposit / open_price[:-1][step_ok]
    step_profit[step_ok] = np.diff(open_price)[step_ok] * step_units
    return step_profit
