"""Train a one-layer PyTorch strategy model on MA features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import Logger
from ml.selflearn_dataset import TRAIN_FRAC, DatasetMeta, build_dataset
from ml.visualization import PredictionVisualizer
from loguru import logger
from tqdm import tqdm

AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME = "prev_pred_label"
AUTOREGRESSIVE_PREV_DRAWDOWN_FEATURE_NAME = "prev_drawdown"
AUTOREGRESSIVE_FEATURE_NAMES = [
    AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME,
    AUTOREGRESSIVE_PREV_DRAWDOWN_FEATURE_NAME,
]
AUTOREGRESSIVE_INITIAL_LABEL = 0.0
AUTOREGRESSIVE_INITIAL_DRAWDOWN = 0.0

Logger(
    log_dir=os.environ.get("ML_LOG_DIR", str(REPO_ROOT / "logs")),
    log_level=os.environ.get("ML_LOG_LEVEL", "INFO"),
).initialize(
    decision_maker="macross_ml",
    symbol="BTCUSDT",
    period="M60",
    clear_logs=False,
)

class OneLayerClassifier(nn.Module):
    def __init__(self, in_features: int, logit_clip: float = 6.0):
        super().__init__()
        self.hidden = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features, 1)
        self.logit_clip = logit_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        # x = self.logit_clip * torch.tanh(x / self.logit_clip)
        return torch.tanh(x)

@dataclass
class TrainArtifacts:
    """Saved training outputs for `SelfLearn.save()` (single, multi-run, or CV result handle)."""

    mode: Literal["single", "multi", "cv"]
    X_shape: np.ndarray | None = None
    timestamps_train: np.ndarray | None = None
    timestamps_test: np.ndarray | None = None
    open_price_train: np.ndarray | None = None
    open_price_test: np.ndarray | None = None
    train_info: dict | None = None
    train_history: dict | None = None
    model: OneLayerClassifier | None = None
    metrics: dict | None = None
    y_pr_train: np.ndarray | None = None
    y_pr_test: np.ndarray | None = None
    train_strategy_step_profit: np.ndarray | None = None
    test_strategy_step_profit: np.ndarray | None = None
    result: dict | None = None
    run_train_strategy_cum_profit: list[np.ndarray] | None = None
    run_test_strategy_cum_profit: list[np.ndarray] | None = None
    run_timestamps_train_steps: list[np.ndarray] | None = None
    run_timestamps_test_steps: list[np.ndarray] | None = None
    open_price: np.ndarray | None = None
    cv_test_strategy_step_profit: list[np.ndarray] | None = None


@dataclass(frozen=True)
class TimeSeriesSegments:
    """Independent, strictly-increasing valid ranges in a flat dataset."""

    segments: list[tuple[int, int]]
    valid_rows: np.ndarray


def resolve_device() -> torch.device:
    requested = os.environ.get("DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda"}:
        logger.warning(f"Unsupported DEVICE='{requested}', falling back to 'cpu'")
        requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE='cuda' requested but CUDA is unavailable, falling back to 'cpu'")
        requested = "cpu"
    return torch.device(requested)


def calendar_years_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Per-row calendar year as int64 (numpy/pandas datetime64 bars)."""
    return pd.DatetimeIndex(np.asarray(timestamps)).year.to_numpy(dtype=np.int64)


def _cv_fold_stats(values: list[float]) -> dict:
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


def _build_valid_panel_mask(X: np.ndarray, open_price: np.ndarray) -> np.ndarray:
    """Build a validity mask for a panel of data."""
    return np.isfinite(open_price) & np.all(np.isfinite(X), axis=2)


def _default_segments(n_rows: int) -> list[tuple[int, int]]:
    return [(0, int(n_rows))] if n_rows > 0 else []


def _compute_step_profit_with_boundaries(
    timestamps: np.ndarray,
    open_price: np.ndarray,
    direction: np.ndarray,
    deposit_multp: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """
    Per-step pnl with boundary handling.

    Returns length n-1 vector; invalid transitions or sequence boundaries are 0.
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
            open_change[i, step_ok] * direction[i, :n_steps][step_ok] * deposit_multp[i, :n_steps][step_ok]
        )
    return step_profit


def _rollout_train_timesteps(model: OneLayerClassifier,
                             X_tensor: torch.Tensor,
                             open_price_tensor: torch.Tensor,
                             deposit_multp_tensor: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    bsz, tlen, _ = X_tensor.shape
    device = X_tensor.device
    dtype = X_tensor.dtype
    step_pnl_parts: list[torch.Tensor] = []
    predicts: torch.Tensor = torch.zeros((bsz, tlen), dtype=dtype, device=device)
    prev_state = torch.zeros((bsz, 2), dtype=dtype, device=device)
    for t in range(tlen):
        valid_t = torch.isfinite(open_price_tensor[:, t]) & torch.all(
            torch.isfinite(X_tensor[:, t, :]), dim=1
        )
        if not torch.any(valid_t):
            # No usable rows this timestep: reset hidden autoregressive state.
            prev_state[:] = 0.0
            continue
        x_t = X_tensor[:, t, :][valid_t]
        prev_t = prev_state[valid_t]
        step_input = torch.cat((x_t, prev_t), dim=1)
        predicts_tensor = model(step_input).squeeze(-1)
        predicts[valid_t, t] = predicts_tensor.detach()
        # Update autoregressive state only for active batch elements.
        prev_state[valid_t, 0] = predicts_tensor
        if t < tlen - 1:
            valid_next = torch.isfinite(open_price_tensor[:, t + 1]) & torch.all(
                torch.isfinite(X_tensor[:, t + 1, :]), dim=1
            )
            active = valid_t & valid_next
            if torch.any(active):
                open_change = (
                    open_price_tensor[active, t + 1] - open_price_tensor[active, t]
                )
                dir_active = prev_state[active, 0]
                step_pnl = open_change * dir_active * deposit_multp_tensor[active, t]
                step_pnl_parts.append(step_pnl)
        # Reset states for currently invalid rows (dynamic batch reduction).
        prev_state[~valid_t] = 0.0
    return predicts.cpu().numpy(), step_pnl_parts


def train_classifier(X_train: np.ndarray, 
                     open_price_train: np.ndarray, 
                     deposit_multp: np.ndarray,
                     device: torch.device,
                     deposit: float,
                     bar_description: str = "Training",
                     segments: list[tuple[int, int]] | None = None) -> tuple[OneLayerClassifier, dict, dict]:
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (B,T,F), got {X_train.shape}")
    bsz, tlen, feat_n = X_train.shape
    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price_train.astype(np.float32)).to(device)
    deposit_multp_tensor = torch.tensor(deposit_multp, dtype=torch.float32, device=device)
    segments = segments if segments is not None else _default_segments(tlen)
    if not segments:
        raise ValueError("No valid segments available for training")

    model = OneLayerClassifier(in_features=feat_n + len(AUTOREGRESSIVE_FEATURE_NAMES), 
                               logit_clip=float(os.environ["LOGIT_CLIP"])).to(device)
    learning_rate = float(os.environ["LR"])
    num_epochs = int(os.environ["EPOCHS"])
    drawdown_lambda = float(os.environ["DRAWDOWN_LAMBDA"])
    l2_lambda = float(os.environ.get("L2_LAMBDA", "0"))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_value = 0.0
    final_profit_value = 0.0
    loss_history: list[float] = []
    profit_history: list[float] = []
    grad_norm_history: list[float] = []
    weight_norm_history: list[float] = []
    weight_norm_change_history: list[float] = []
    l2_penalty_history: list[float] = []
    use_l2 = l2_lambda != 0.0
    prev_weight_norm = 0.0
    progress_bar = tqdm(range(num_epochs), desc=bar_description)
    for _ in progress_bar:
        optimizer.zero_grad()
        _, step_pnl_parts = _rollout_train_timesteps(
            model,
            X_tensor,
            open_price_tensor,
            deposit_multp_tensor,
        )

        sequence_profit = torch.sum(torch.cat(step_pnl_parts))
        relative_outperformance = sequence_profit / deposit / bsz * 100
        if use_l2:
            l2_penalty = sum((p * p).sum() for p in model.parameters())
        else:
            l2_penalty = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
        loss = (-relative_outperformance + l2_lambda * l2_penalty)
        postfix: dict[str, str] = {
            "loss": f"{loss.item():.3f}",
            "rel_outperf": f"{relative_outperformance.item():.3f}",
        }
        if use_l2:
            postfix["l2"] = f"{l2_penalty.item():.3f}"
        progress_bar.set_postfix(**postfix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(os.environ["GRAD_MAX_NORM"]))
        grad_sq_sum = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq_sum += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = grad_sq_sum**0.5
        optimizer.step()
        weight_sq_sum = 0.0
        for param in model.parameters():
            weight_sq_sum += float(torch.sum(param.detach() ** 2).item())
        weight_norm = weight_sq_sum**0.5
        weight_norm_change = weight_norm - prev_weight_norm
        prev_weight_norm = weight_norm
        loss_value = float(loss.item())
        final_profit_value = float(relative_outperformance.item())
        loss_history.append(loss_value)
        profit_history.append(final_profit_value)
        grad_norm_history.append(grad_norm)
        weight_norm_history.append(weight_norm)
        weight_norm_change_history.append(weight_norm_change)
        l2_penalty_history.append(float(l2_penalty.item()))

    train_info = {
        "optimizer": "Adam",
        "device": str(device),
        "loss": "neg_relative_outperformance_plus_drawdown_penalty_plus_optional_l2",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "drawdown_lambda": drawdown_lambda,
        "l2_lambda": l2_lambda,
        "autoregressive_prev_label": True,
        "autoregressive_prev_drawdown": True,
        "autoregressive_feature_name": AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME,
        "autoregressive_feature_names": AUTOREGRESSIVE_FEATURE_NAMES,
        "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
        "autoregressive_initial_prev_drawdown": AUTOREGRESSIVE_INITIAL_DRAWDOWN,
        "final_train_loss": loss_value,
        "final_train_profit": final_profit_value,
        "final_l2_penalty": l2_penalty_history[-1] if l2_penalty_history else 0.0,
        "final_grad_norm": grad_norm_history[-1] if grad_norm_history else 0.0,
        "final_weight_norm": weight_norm_history[-1] if weight_norm_history else 0.0,
    }
    train_history = {
        "loss": loss_history,
        "final_profit": profit_history,
        "grad_norm": grad_norm_history,
        "weight_norm": weight_norm_history,
        "weight_norm_change": weight_norm_change_history,
        "l2_penalty": l2_penalty_history,
    }
    return model, train_info, train_history


def predict_direction(
    model: OneLayerClassifier,
    X: np.ndarray,
    open_price: np.ndarray,
    deposit_multp: np.ndarray,
    device: torch.device,
    score_threshold: float = 0.,
) -> np.ndarray:
    bsz, tlen, _ = X.shape
    score_thresholds = np.ones((bsz, 1)) * score_threshold
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price.astype(np.float32)).to(device)
    deposit_multp_tensor = torch.from_numpy(deposit_multp.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        predicts, _ = _rollout_train_timesteps(
            model,
            X_tensor,
            open_price_tensor,
            deposit_multp_tensor,
        )
        position_dirs = np.where(predicts >= score_thresholds, 1, -1).astype(np.int64)
    return position_dirs


class SelfLearn:
    def __init__(self, config: str, output_dir: Path, deposit: float = 1000.0):
        self.config_path = config
        self.output_dir = output_dir
        self.deposit = deposit
        self.device = resolve_device()
        self._artifacts: TrainArtifacts | None = None
        self._meta: DatasetMeta | None = None

    def _train_from_window(
        self,
        X: np.ndarray,
        timestamps: np.ndarray,
        open_price: np.ndarray,
        train_size: int | None = None,
    ) -> TrainArtifacts:
        bsz, tlen, n_features = X.shape
        if train_size is None:
            train_size = int(np.round(tlen * TRAIN_FRAC))
        train_size = max(1, min(train_size, tlen - 1))
        test_size = tlen - train_size

        X_train = X[:, :train_size, :]
        X_test = X[:, train_size:, :]
        timestamps_train = timestamps[:, :train_size]
        timestamps_test = timestamps[:, train_size:]
        open_price_train = open_price[:, :train_size]
        open_price_test = open_price[:, train_size:]

        valid_train_panel = _build_valid_panel_mask(X_train, open_price_train)
        valid_test_panel = _build_valid_panel_mask(X_test, open_price_test)

        if not np.any(valid_train_panel):
            raise ValueError("No valid training rows after NaN filtering")
        deposit_multp = np.zeros_like(open_price, dtype=np.float64)
        pos_mask = np.isfinite(open_price) & (open_price > 0.0)
        deposit_multp[pos_mask] = self.deposit / open_price[pos_mask]
        deposit_multp_train = deposit_multp[:, :train_size]
        deposit_multp_test = deposit_multp[:, train_size:]

        model, train_info, train_history = train_classifier(
            X_train,
            open_price_train,
            deposit_multp=deposit_multp_train,
            device=self.device,
            deposit=self.deposit,
            bar_description="Training: ",
        )
        y_pr_train_panel = predict_direction(
            model,
            X_train,
            open_price_train,
            deposit_multp_train,
            device=self.device,
        )
        y_pr_test_panel = predict_direction(
            model,
            X_test,
            open_price_test,
            deposit_multp_test,
            device=self.device,
        )

        train_strategy_step_profit = _compute_step_profit_with_boundaries(
            timestamps=timestamps_train,
            open_price=open_price_train,
            direction=y_pr_train_panel,
            deposit_multp=deposit_multp_train,
            valid_rows=valid_train_panel,
        )
        test_strategy_step_profit = _compute_step_profit_with_boundaries(
            timestamps=timestamps_test,
            open_price=open_price_test,
            direction=y_pr_test_panel,
            deposit_multp=deposit_multp_test,
            valid_rows=valid_test_panel,
        )

        metrics = {
            "dataset": {
                "aligned_rows": int(bsz * tlen),
                "batch_size": int(bsz),
                "timesteps": int(tlen),
                "train_size": train_size,
                "test_size": test_size,
                "valid_rows_train": np.sum(valid_train_panel, axis=1).tolist(),
                "valid_rows_test": np.sum(valid_test_panel, axis=1).tolist(),
            },
            "training": train_info,
        }

        return TrainArtifacts(
            mode="single",
            X_shape=X.shape,
            timestamps_train=timestamps_train,
            timestamps_test=timestamps_test,
            open_price_train=open_price_train,
            open_price_test=open_price_test,
            train_info=train_info,
            train_history=train_history,
            model=model,
            metrics=metrics,
            y_pr_train=y_pr_train_panel,
            y_pr_test=y_pr_test_panel,
            train_strategy_step_profit=train_strategy_step_profit,
            test_strategy_step_profit=test_strategy_step_profit,
        )

    def train(self) -> dict:
        X, meta = build_dataset(self.config_path)
        timestamps = np.asarray(meta.timestamps)
        open_price = np.asarray(meta.open_price, dtype=np.float64)
        self._meta = meta
        self._artifacts = self._train_from_window(
            X=X,
            timestamps=timestamps,
            open_price=open_price,
        )
        assert self._artifacts.metrics is not None
        return self._artifacts.metrics

    def train_multip(self, runs: int = 5, max_cut_frac: float = 0.1, seed: int | None = None) -> dict:
        if runs < 1:
            raise ValueError("runs must be >= 1")
        if not 0.0 <= max_cut_frac < 0.5:
            raise ValueError("max_cut_frac must be in [0.0, 0.5)")

        X, meta = build_dataset(self.config_path)
        self._meta = meta
        timestamps = np.asarray(meta.timestamps)
        open_price = np.asarray(meta.open_price, dtype=np.float64)
        X_panel, ts_panel, open_panel = _ensure_panel_from_flat_dataset(X, timestamps, open_price)
        _, n_times, _ = X_panel.shape
        if n_times < 4:
            raise ValueError("Not enough timesteps for train_multip; need at least 4")

        full_train_size = int(np.round(n_times * TRAIN_FRAC))
        full_train_size = max(1, min(full_train_size, n_times - 1))
        fixed_test_start_idx = full_train_size
        train_pool_len = full_train_size

        rng = np.random.default_rng(seed)
        max_cut = int(np.floor(train_pool_len * max_cut_frac))
        min_window_rows = 4

        run_metrics: list[dict] = []
        strategy_profit_values: list[float] = []
        final_loss_values: list[float] = []
        run_train_strategy_cum_profit: list[np.ndarray] = []
        run_test_strategy_cum_profit: list[np.ndarray] = []
        run_timestamps_train_steps: list[np.ndarray] = []
        run_timestamps_test_steps: list[np.ndarray] = []

        for run_idx in range(runs):
            left_cut = int(rng.integers(0, max_cut + 1)) if max_cut > 0 else 0
            right_cut = int(rng.integers(0, max_cut + 1)) if max_cut > 0 else 0
            train_global_start = left_cut
            train_global_end = train_pool_len - right_cut
            if train_global_end - train_global_start < min_window_rows:
                train_global_start = 0
                train_global_end = train_pool_len

            X_train_seg = X_panel[:, train_global_start:train_global_end, :]
            X_test_seg = X_panel[:, fixed_test_start_idx:n_times, :]
            X_window = np.concatenate([X_train_seg, X_test_seg], axis=1)
            timestamps_window = np.concatenate(
                [ts_panel[:, train_global_start:train_global_end], ts_panel[:, fixed_test_start_idx:n_times]],
                axis=1,
            )
            open_price_window = np.concatenate(
                [open_panel[:, train_global_start:train_global_end], open_panel[:, fixed_test_start_idx:n_times]],
                axis=1,
            )
            train_rows = int(X_train_seg.shape[1])

            artifacts = self._train_from_window(
                X=X_window,
                timestamps=timestamps_window,
                open_price=open_price_window,
                train_size=train_rows,
            )
            metrics = artifacts.metrics
            assert metrics is not None
            strategy_profit = float(metrics["profit"]["strategy_final_profit"])
            final_loss = float(metrics["training"]["final_train_loss"])
            strategy_profit_values.append(strategy_profit)
            final_loss_values.append(final_loss)
            tr_step = artifacts.train_strategy_step_profit
            te_step = artifacts.test_strategy_step_profit
            ts_tr = artifacts.timestamps_train
            ts_te = artifacts.timestamps_test
            if tr_step.size:
                run_train_strategy_cum_profit.append(np.cumsum(tr_step))
                run_timestamps_train_steps.append(np.asarray(ts_tr[:-1]))
            else:
                run_train_strategy_cum_profit.append(np.array([], dtype=np.float64))
                run_timestamps_train_steps.append(np.array([], dtype=np.asarray(ts_tr).dtype))
            if te_step.size:
                run_test_strategy_cum_profit.append(np.cumsum(te_step))
                run_timestamps_test_steps.append(np.asarray(ts_te[:-1]))
            else:
                run_test_strategy_cum_profit.append(np.array([], dtype=np.float64))
                run_timestamps_test_steps.append(np.array([], dtype=np.asarray(ts_te).dtype))
            run_metrics.append(
                {
                    "run_idx": run_idx,
                    "train_global_start_idx": train_global_start,
                    "train_global_end_idx": train_global_end,
                    "fixed_test_start_idx": fixed_test_start_idx,
                    "left_cut": left_cut,
                    "right_cut": right_cut,
                    "metrics": metrics,
                }
            )

        strategy_profit_arr = np.asarray(strategy_profit_values, dtype=np.float64)
        final_loss_arr = np.asarray(final_loss_values, dtype=np.float64)

        def _stats(values: np.ndarray) -> dict:
            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            return {
                "mean": mean_value,
                "std": std_value,
                "cv_abs_mean": float(std_value / max(abs(mean_value), 1e-6)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        stability = {
            "runs": runs,
            "seed": seed,
            "max_cut_frac": max_cut_frac,
            "base_aligned_rows": int(X_panel.shape[0] * X_panel.shape[1]),
            "fixed_test_start_idx": int(fixed_test_start_idx),
            "fixed_test_rows": int(n_times - fixed_test_start_idx),
            "train_pool_rows": int(train_pool_len),
            "strategy_final_profit": _stats(strategy_profit_arr),
            "final_train_loss": _stats(final_loss_arr),
        }
        result = {
            "stability": stability,
            "runs": run_metrics,
        }
        self._artifacts = TrainArtifacts(
            mode="multi",
            result=result,
            run_train_strategy_cum_profit=run_train_strategy_cum_profit,
            run_test_strategy_cum_profit=run_test_strategy_cum_profit,
            run_timestamps_train_steps=run_timestamps_train_steps,
            run_timestamps_test_steps=run_timestamps_test_steps,
            open_price=open_panel.reshape(-1),
        )
        return result

    def cross_validation(self) -> dict:
        X, meta = build_dataset(self.config_path)
        self._meta = meta
        timestamps = np.asarray(meta.timestamps)
        open_price = np.asarray(meta.open_price, dtype=np.float64)
        n = int(X_panel.shape[0] * X_panel.shape[1])
        n_times = int(X_panel.shape[1])

        if n_times < 2:
            logger.warning("Not enough timesteps for cross_validation; need at least 2")
            result = {
                "folds": [],
                "summary": {
                    "n_folds": 0,
                    "test_years": [],
                    "test_strategy_profit_sum": _cv_fold_stats([]),
                    "base_aligned_rows": n,
                },
            }
            self._artifacts = TrainArtifacts(
                mode="cv",
                result=result,
                open_price=open_panel.reshape(-1),
                run_timestamps_test_steps=[],
                cv_test_strategy_step_profit=[],
            )
            return result

        years = calendar_years_from_timestamps(ts_panel[0])
        unique_years = np.unique(years)

        folds: list[dict] = []
        test_strategy_values: list[float] = []
        cv_ts_steps: list[np.ndarray] = []
        cv_strat_steps: list[np.ndarray] = []

        for Y in unique_years:
            logger.info(f"Training for test year {int(Y)}")
            idx_train = np.flatnonzero(years != Y)
            idx_test = np.flatnonzero(years == Y)
            if idx_train.size < 1 or idx_test.size < 1:
                logger.warning(
                    f"Skipping fold test_year={int(Y)}: need at least one train and one test row"
                )
                continue

            X_window = np.concatenate([X_panel[:, idx_train, :], X_panel[:, idx_test, :]], axis=1)
            timestamps_window = np.concatenate(
                [ts_panel[:, idx_train], ts_panel[:, idx_test]], axis=1
            )
            open_price_window = np.concatenate(
                [open_panel[:, idx_train], open_panel[:, idx_test]], axis=1
            )
            train_rows = int(idx_train.size)

            artifacts = self._train_from_window(
                X=X_window,
                timestamps=timestamps_window,
                open_price=open_price_window,
                train_size=train_rows,
            )
            assert artifacts.metrics is not None
            te_step = artifacts.test_strategy_step_profit
            test_strat_sum = float(np.sum(te_step)) if te_step is not None and te_step.size else 0.0

            ts_te = artifacts.timestamps_test
            assert ts_te is not None and te_step is not None
            if te_step.size:
                cv_ts_steps.append(np.asarray(ts_te[:-1]))
                cv_strat_steps.append(np.asarray(te_step, dtype=np.float64))
            else:
                cv_ts_steps.append(np.array([], dtype=np.asarray(ts_te).dtype))
                cv_strat_steps.append(np.array([], dtype=np.float64))

            folds.append(
                {
                    "test_year": int(Y),
                    "train_rows": train_rows,
                    "test_rows": int(idx_test.size),
                    "metrics": artifacts.metrics,
                    "test_strategy_profit_sum": test_strat_sum,
                }
            )
            test_strategy_values.append(test_strat_sum)

        if not folds:
            logger.warning(
                "cross_validation produced no valid folds (e.g. single calendar year in data)"
            )

        summary = {
            "n_folds": len(folds),
            "test_years": [f["test_year"] for f in folds],
            "test_strategy_profit_sum": _cv_fold_stats(test_strategy_values),
            "base_aligned_rows": n,
        }
        result = {"folds": folds, "summary": summary}
        self._artifacts = TrainArtifacts(
            mode="cv",
            result=result,
            open_price=open_panel.reshape(-1),
            run_timestamps_test_steps=cv_ts_steps,
            cv_test_strategy_step_profit=cv_strat_steps,
        )
        return result

    def save(self) -> dict:
        if self._artifacts is None:
            raise RuntimeError("Nothing to save. Run train() before save().")

        mode = self._artifacts.mode
        if mode == "single":
            if self._meta is None:
                raise RuntimeError("Dataset metadata missing. Run train() before save().")
            return self._save_single()
        if mode == "multi":
            return self._save_multi()
        if mode == "cv":
            return self._save_cv()
        raise RuntimeError(f"Unsupported save mode: {mode}")

    def _save_multi(self) -> dict:
        a = self._artifacts
        assert a is not None
        result = a.result
        run_train_strategy_cum_profit = a.run_train_strategy_cum_profit
        run_test_strategy_cum_profit = a.run_test_strategy_cum_profit
        run_timestamps_train_steps = a.run_timestamps_train_steps
        run_timestamps_test_steps = a.run_timestamps_test_steps
        open_price = a.open_price

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stability_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        logger.info(f"Stability metrics saved to: {self.output_dir / 'stability_metrics.json'}")
        visualizer = PredictionVisualizer(
            deposit=self.deposit,
            open_price_train=open_price,
            open_price_test=open_price,
        )
        visualizer.save_multi_run_profit_plot(
            run_train_strategy_cum_profit=run_train_strategy_cum_profit,
            run_test_strategy_cum_profit=run_test_strategy_cum_profit,
            run_timestamps_train_steps=run_timestamps_train_steps,
            run_timestamps_test_steps=run_timestamps_test_steps,
            output_path=self.output_dir / "profit_multi_run.png",
        )
        logger.info(f"Multi-run profit chart saved to: {self.output_dir / 'profit_multi_run.png'}")
        return result

    def _save_cv(self) -> dict:
        a = self._artifacts
        assert a is not None
        result = a.result
        assert result is not None
        fold_ts = a.run_timestamps_test_steps or []
        strat_steps = a.cv_test_strategy_step_profit or []
        open_price = a.open_price

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_json = self.output_dir / "cross_validation_metrics.json"
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        logger.info(f"Cross-validation metrics saved to: {out_json}")

        visualizer = PredictionVisualizer(
            deposit=self.deposit,
            open_price_train=open_price,
            open_price_test=open_price,
        )
        out_plot = self.output_dir / "profit_cross_validation.png"
        visualizer.save_cv_chained_oos_profit_plot(
            fold_timestamps_test_steps=fold_ts,
            fold_strategy_step_profit=strat_steps,
            output_path=out_plot,
        )
        logger.info(f"Cross-validation profit chart saved to: {out_plot}")
        return result

    def _save_single(self) -> dict:
        a = self._artifacts
        meta = self._meta
        assert a is not None and meta is not None
        X_shape = a.X_shape
        timestamps_train = a.timestamps_train
        timestamps_test = a.timestamps_test
        open_price_train = a.open_price_train
        open_price_test = a.open_price_test
        train_info = a.train_info
        train_history = a.train_history
        model = a.model
        metrics = a.metrics
        y_pr_train = a.y_pr_train
        y_pr_test = a.y_pr_test
        train_strategy_step_profit = a.train_strategy_step_profit
        test_strategy_step_profit = a.test_strategy_step_profit

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_features": int(X_shape[1] + len(AUTOREGRESSIVE_FEATURE_NAMES)),
                "base_feature_count": int(X_shape[1]),
                "autoregressive_prev_label": True,
                "autoregressive_prev_drawdown": True,
                "autoregressive_feature_name": AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME,
                "autoregressive_feature_names": AUTOREGRESSIVE_FEATURE_NAMES,
                "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
                "autoregressive_initial_prev_drawdown": AUTOREGRESSIVE_INITIAL_DRAWDOWN,
                "train_info": train_info,
            },
            self.output_dir / "model.pt",
        )
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        visualizer = PredictionVisualizer(
            deposit=self.deposit,
            open_price_train=open_price_train,
            open_price_test=open_price_test,
        )
        for i, symbol in enumerate(meta.symbols):
            visualizer.save_strategy_train_test_profit_plot(
                timestamps_train=timestamps_train[i][:-1],
                strategy_cum_train=np.cumsum(train_strategy_step_profit[i]) if train_strategy_step_profit[i].size else np.array([], dtype=np.float64),
                timestamps_test=timestamps_test[i][:-1],
                strategy_cum_test=np.cumsum(test_strategy_step_profit[i]) if test_strategy_step_profit[i].size else np.array([], dtype=np.float64),
                pred_sign_train=y_pr_train[i][:-1],
                pred_sign_test=y_pr_test[i][:-1],
                output_path=self.output_dir / f"strategy_profit_train_test_{symbol.ticker}.png",
            )
        logger.info(f"Strategy profit plot saved to: {self.output_dir / 'strategy_profit_train_test.png'}")
        visualizer.save_loss_change_plot(
            loss_values=np.asarray(train_history["loss"], dtype=np.float64),
            profit_values=np.asarray(train_history["final_profit"], dtype=np.float64),
            output_path=self.output_dir / "loss_change_train.png",
        )
        logger.info(f"Loss change plot saved to: {self.output_dir / 'loss_change_train.png'}")
        visualizer.save_gradient_change_plot(
            grad_norm_values=np.asarray(train_history["grad_norm"], dtype=np.float64),
            output_path=self.output_dir / "gradient_change_train.png",
        )
        logger.info(f"Gradient norm plot saved to: {self.output_dir / 'gradient_change_train.png'}")
        visualizer.save_weight_norm_change_plot(
            weight_norm_values=np.asarray(train_history["weight_norm"], dtype=np.float64),
            weight_change_values=np.asarray(train_history["weight_norm_change"], dtype=np.float64),
            output_path=self.output_dir / "weights_norm_change_train.png",
        )
        logger.info(f"Weights norm change plot saved to: {self.output_dir / 'weights_norm_change_train.png'}")
        schema = {
            "feature_names": meta.feature_names + AUTOREGRESSIVE_FEATURE_NAMES,
            "model_type": "OneLayerTorchBinaryClassifier",
            "model_params": {
                "in_features": int(X_shape[1] + len(AUTOREGRESSIVE_FEATURE_NAMES)),
                "base_feature_count": int(X_shape[1]),
                "out_features": 1,
                "threshold": 0.5,
                "autoregressive_prev_label": True,
                "autoregressive_prev_drawdown": True,
                "autoregressive_feature_name": AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME,
                "autoregressive_feature_names": AUTOREGRESSIVE_FEATURE_NAMES,
                "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
                "autoregressive_initial_prev_drawdown": AUTOREGRESSIVE_INITIAL_DRAWDOWN,
                "train_info": train_info,
            },
        }
        (self.output_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("single", "multi", "cross"),
        default="single",
        help="Training mode: single train/test split, multi-run stability, or cross-validation",
    )
    args = parser.parse_args()

    if os.environ.get("ML_OUTPUT_DIR") is None:
        logger.error("ML_OUTPUT_DIR is not set")
        return 1
    output_dir = Path(os.environ.get("ML_OUTPUT_DIR")).resolve()
    self_learner = SelfLearn(config=os.environ.get("CONFIG"), output_dir=output_dir, deposit=float(os.environ.get("DEPOSIT")))
    if args.mode == "single":
        metrics = self_learner.train()
    elif args.mode == "multi":
        metrics = self_learner.train_multip()
    else:
        metrics = self_learner.cross_validation()
    metrics = self_learner.save()
    logger.info(f"One-layer PyTorch classifier artifacts saved to: {output_dir}")
    if "profit" in metrics:
        logger.info(f"Strategy final profit: {metrics['profit']['strategy_final_profit']:.6f}")
    elif "stability" in metrics:
        logger.info(
            "Multi-run strategy final profit mean/std: "
            f"{metrics['stability']['strategy_final_profit']['mean']:.6f}/"
            f"{metrics['stability']['strategy_final_profit']['std']:.6f}"
        )
    elif "summary" in metrics:
        logger.info(
            "Cross-validation test strategy profit mean/std: "
            f"{metrics['summary']['test_strategy_profit_sum']['mean']:.6f}/"
            f"{metrics['summary']['test_strategy_profit_sum']['std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
