from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import PyConfig


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
    return pd.DatetimeIndex(np.asarray(timestamps)).year.to_numpy(dtype=np.int64)


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
) -> PackedYearlyTrainingData:
    """Pack (previous year warm-up, current year target) sequences into one batch."""
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (B,T,F), got {X_train.shape}")

    bsz, _, n_features = X_train.shape
    valid_panel = build_valid_panel_mask(X_train, open_price_train) & valid_timestamp_mask(
        timestamps_train
    )
    sequences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    sequence_meta: list[dict] = []
    skipped_first_years: list[dict] = []
    target_years: set[int] = set()
    warmup_years: set[int] = set()

    for symbol_idx in range(bsz):
        valid_idx = np.flatnonzero(valid_panel[symbol_idx])
        if valid_idx.size == 0:
            continue

        valid_ts = timestamps_train[symbol_idx, valid_idx]
        valid_years = pd.DatetimeIndex(valid_ts).year.to_numpy(dtype=np.int64)
        unique_years = sorted(int(year) for year in np.unique(valid_years))
        if unique_years:
            skipped_first_years.append(
                {"symbol_index": symbol_idx, "year": int(unique_years[0])}
            )

        year_to_idx = {
            year: valid_idx[valid_years == year]
            for year in unique_years
        }
        for target_year in unique_years[1:]:
            warmup_year = target_year - 1
            if warmup_year not in year_to_idx:
                continue
            warmup_idx = year_to_idx[warmup_year]
            target_idx = year_to_idx[target_year]
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
            warmup_years.add(int(warmup_year))
            target_years.add(int(target_year))
            sequence_meta.append(
                {
                    "symbol_index": symbol_idx,
                    "warmup_year": int(warmup_year),
                    "target_year": int(target_year),
                    "warmup_rows": int(warmup_idx.size),
                    "target_rows": int(target_idx.size),
                }
            )

    if not sequences:
        raise ValueError(
            "No yearly training sequences with a previous calendar year warm-up "
            "and at least two target rows"
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

    metadata = {
        "train_batching": "calendar_year_with_previous_year_warmup",
        "loss_reduction": "mean_per_year_sequence",
        "original_batch_size": int(bsz),
        "original_timesteps": int(X_train.shape[1]),
        "packed_batch_size": int(packed_bsz),
        "packed_timesteps": int(max_len),
        "target_years": sorted(target_years),
        "warmup_years": sorted(warmup_years),
        "skipped_first_years": skipped_first_years,
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
    """Buy-and-hold PnL for one symbol, using first valid open as entry."""
    n = int(open_price.shape[0])
    if n <= 1:
        return np.array([], dtype=np.float64)

    step_profit = np.zeros(n - 1, dtype=np.float64)
    valid_rows = np.isfinite(open_price) & (open_price > 0.0)
    valid_idx = np.flatnonzero(valid_rows)
    if valid_idx.size == 0:
        return step_profit

    units = deposit / float(open_price[valid_idx[0]])
    step_ok = valid_rows[:-1] & valid_rows[1:] & (timestamps[1:] > timestamps[:-1])
    step_profit[step_ok] = np.diff(open_price)[step_ok] * units
    return step_profit
