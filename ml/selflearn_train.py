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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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

class GruPolicy(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 16, logit_clip: float = 6.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(in_features, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.logit_clip = logit_clip

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(x, hidden)
        x = self.out(hidden)
        # x = self.logit_clip * torch.tanh(x / self.logit_clip)
        return torch.tanh(x), hidden

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
    training_checkpoint: dict | None = None
    model: GruPolicy | None = None
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


def resolve_resume_checkpoint(
    output_dir: Path,
    cli_path: str | None = None,
) -> Path | None:
    """Resolve checkpoint path from CLI, RESUME_CHECKPOINT, or RESUME=1 -> output_dir/model.pt."""
    if cli_path is not None:
        path = (output_dir / "model.pt") if cli_path == "" else Path(cli_path).expanduser()
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    explicit = os.environ.get("RESUME_CHECKPOINT", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    resume = os.environ.get("RESUME", "").strip().lower()
    if resume in {"1", "true", "yes", "on"}:
        path = (output_dir / "model.pt").resolve()
        if path.is_file():
            return path
        logger.warning(f"RESUME enabled but checkpoint missing: {path}")
    return None


def load_training_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {path}")
    return checkpoint


def _history_list(checkpoint: dict | None, key: str) -> list[float]:
    if checkpoint is None:
        return []
    train_history = checkpoint.get("train_history")
    if not isinstance(train_history, dict):
        return []
    values = train_history.get(key, [])
    return [float(v) for v in values]


def _build_model_from_checkpoint(
    checkpoint: dict,
    feat_n: int,
    device: torch.device,
    logit_clip: float,
) -> GruPolicy:
    expected_in_features = feat_n + len(AUTOREGRESSIVE_FEATURE_NAMES)
    in_features = int(checkpoint.get("in_features", expected_in_features))
    if in_features != expected_in_features:
        raise ValueError(
            f"Checkpoint in_features={in_features} does not match dataset "
            f"({expected_in_features})"
        )
    hidden_size = int(checkpoint.get("gru_hidden_size", os.environ.get("GRU_HIDDEN_SIZE", "16")))
    model = GruPolicy(
        in_features=in_features,
        hidden_size=hidden_size,
        logit_clip=logit_clip,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


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


def _compute_step_profit_with_boundaries(timestamps: np.ndarray,
                                         open_price: np.ndarray,
                                         direction: np.ndarray,
                                         deposit_multp: np.ndarray,
                                         valid_rows: np.ndarray
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


def _compute_buy_hold_step_profit(
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


def _compute_benchmark_step_pnl_parts(
    X_tensor: torch.Tensor,
    open_price_tensor: torch.Tensor,
    deposit_multp_tensor: torch.Tensor,
) -> list[torch.Tensor]:
    """Buy-and-hold (+1 direction) PnL using the same step logic as training rollout."""
    bsz, tlen, _ = X_tensor.shape
    step_pnl_parts: list[torch.Tensor] = []
    for t in range(tlen):
        valid_t = torch.isfinite(open_price_tensor[:, t]) & torch.all(
            torch.isfinite(X_tensor[:, t, :]), dim=1
        )
        if t < tlen - 1:
            valid_next = torch.isfinite(open_price_tensor[:, t + 1]) & torch.all(
                torch.isfinite(X_tensor[:, t + 1, :]), dim=1
            )
            active = valid_t & valid_next
            if torch.any(active):
                open_change = (
                    open_price_tensor[active, t + 1] - open_price_tensor[active, t]
                )
                step_pnl_parts.append(open_change * deposit_multp_tensor[active, t])
    return step_pnl_parts


def _compute_soft_deal_crossings(
    predicts: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable proxy for direction flips; zero when direction never changes sign."""
    pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    if not torch.any(pair_valid):
        return torch.zeros((), dtype=predicts.dtype, device=predicts.device)
    crossings = torch.relu(-predicts[:, :-1] * predicts[:, 1:])
    return crossings[pair_valid].sum()


def _compute_deals_penalty(deal_crossings: torch.Tensor, deals_lambda: float) -> torch.Tensor:
    if deals_lambda == 0.0:
        return torch.zeros((), dtype=deal_crossings.dtype, device=deal_crossings.device)
    return deal_crossings.new_tensor(deals_lambda) / (deal_crossings + 1.0)


def _rollout_train_timesteps(model: GruPolicy,
                             X_tensor: torch.Tensor,
                             open_price_tensor: torch.Tensor,
                             deposit_multp_tensor: torch.Tensor,
                             detach_predictions: bool = True,
                             ) -> tuple[torch.Tensor | np.ndarray, list[torch.Tensor], torch.Tensor]:
    bsz, tlen, _ = X_tensor.shape
    device = X_tensor.device
    dtype = X_tensor.dtype
    step_pnl_parts: list[torch.Tensor] = []
    predicts: torch.Tensor = torch.zeros((bsz, tlen), dtype=dtype, device=device)
    valid_mask = torch.zeros((bsz, tlen), dtype=torch.bool, device=device)
    prev_state = torch.zeros((bsz, 2), dtype=dtype, device=device)
    hidden = torch.zeros((bsz, model.hidden_size), dtype=dtype, device=device)
    for t in range(tlen):
        valid_t = torch.isfinite(open_price_tensor[:, t]) & torch.all(
            torch.isfinite(X_tensor[:, t, :]), dim=1
        )
        if not torch.any(valid_t):
            # No usable rows this timestep: reset hidden autoregressive state.
            prev_state = prev_state * 0.0
            hidden = hidden * 0.0
            continue
        valid_mask[:, t] = valid_t
        x_t = X_tensor[:, t, :][valid_t]
        prev_t = prev_state[valid_t]
        step_input = torch.cat((x_t, prev_t), dim=1)
        predicts_tensor, hidden_t = model(step_input, hidden[valid_t])
        predicts_tensor = predicts_tensor.squeeze(-1)
        if detach_predictions:
            predicts[valid_t, t] = predicts_tensor.detach()
        else:
            predicts[valid_t, t] = predicts_tensor
        # Update autoregressive state only for active batch elements.
        next_prev_state = prev_state.clone()
        next_prev_state[valid_t, 0] = predicts_tensor
        prev_state = next_prev_state
        next_hidden = hidden.clone()
        next_hidden[valid_t] = hidden_t
        hidden = next_hidden
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
        prev_state = prev_state * valid_t[:, None].to(dtype)
        hidden = hidden * valid_t[:, None].to(dtype)
    if detach_predictions:
        return predicts.detach().cpu().numpy(), step_pnl_parts, valid_mask
    return predicts, step_pnl_parts, valid_mask


def train_classifier(X_train: np.ndarray, 
                     open_price_train: np.ndarray, 
                     deposit_multp: np.ndarray,
                     device: torch.device,
                     deposit: float,
                     bar_description: str = "Training",
                     segments: list[tuple[int, int]] | None = None,
                     resume_checkpoint: Path | None = None,
                    ) -> tuple[GruPolicy, dict, dict, dict]:
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (B,T,F), got {X_train.shape}")
    bsz, tlen, feat_n = X_train.shape
    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price_train.astype(np.float32)).to(device)
    deposit_multp_tensor = torch.tensor(deposit_multp, dtype=torch.float32, device=device)
    segments = segments if segments is not None else _default_segments(tlen)
    if not segments:
        raise ValueError("No valid segments available for training")

    hidden_size = int(os.environ.get("GRU_HIDDEN_SIZE", "16"))
    logit_clip = float(os.environ["LOGIT_CLIP"])
    learning_rate = float(os.environ["LR"])
    num_epochs = int(os.environ["EPOCHS"])
    drawdown_lambda = float(os.environ["DRAWDOWN_LAMBDA"])
    deals_lambda = float(os.environ.get("DEALS_LAMBDA", "0"))
    lr_restart_period = int(os.environ.get("LR_RESTART_PERIOD", "50"))
    lr_min = float(os.environ.get("LR_MIN", "1e-6"))

    checkpoint: dict | None = None
    if resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(resume_checkpoint, device)
        hidden_size = int(checkpoint.get("gru_hidden_size", hidden_size))
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")

    model = (
        _build_model_from_checkpoint(checkpoint, feat_n, device, logit_clip)
        if checkpoint is not None
        else GruPolicy(
            in_features=feat_n + len(AUTOREGRESSIVE_FEATURE_NAMES),
            hidden_size=hidden_size,
            logit_clip=logit_clip,
        ).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=lr_restart_period,
        T_mult=1,
        eta_min=lr_min,
    )
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("Checkpoint has no optimizer state; starting optimizer fresh")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logger.warning("Checkpoint has no scheduler state; starting scheduler fresh")

    start_epoch = int(checkpoint.get("epochs_completed", 0)) if checkpoint is not None else 0
    prev_weight_norm = float(checkpoint.get("prev_weight_norm", 0.0)) if checkpoint is not None else 0.0
    loss_history: list[float] = _history_list(checkpoint, "loss")
    profit_history: list[float] = _history_list(checkpoint, "final_profit")
    grad_norm_history: list[float] = _history_list(checkpoint, "grad_norm")
    weight_norm_history: list[float] = _history_list(checkpoint, "weight_norm")
    weight_norm_change_history: list[float] = _history_list(checkpoint, "weight_norm_change")
    deals_penalty_history: list[float] = _history_list(checkpoint, "deals_penalty")
    deal_crossings_history: list[float] = _history_list(checkpoint, "deal_crossings")
    strategy_profit_history: list[float] = _history_list(checkpoint, "strategy_profit")
    benchmark_profit_history: list[float] = _history_list(checkpoint, "benchmark_profit")
    if checkpoint is not None and start_epoch != len(loss_history):
        start_epoch = len(loss_history)

    model.train()
    loss_value = 0.0
    final_profit_value = 0.0
    benchmark_step_pnl_parts = _compute_benchmark_step_pnl_parts(
        X_tensor,
        open_price_tensor,
        deposit_multp_tensor,
    )
    if benchmark_step_pnl_parts:
        benchmark_profit_total = torch.sum(torch.cat(benchmark_step_pnl_parts))
    else:
        benchmark_profit_total = torch.zeros((), dtype=X_tensor.dtype, device=device)
    benchmark_profit_pct = benchmark_profit_total / deposit / bsz * 100

    progress_bar = tqdm(
        range(num_epochs),
        desc=bar_description,
        initial=start_epoch,
        total=start_epoch + num_epochs,
    )
    for _ in progress_bar:
        optimizer.zero_grad()
        predicts, step_pnl_parts, valid_mask = _rollout_train_timesteps(
            model,
            X_tensor,
            open_price_tensor,
            deposit_multp_tensor,
            detach_predictions=False,
        )

        sequence_profit = torch.sum(torch.cat(step_pnl_parts))
        strategy_profit_pct = sequence_profit / deposit / bsz * 100
        excess_profit_pct = strategy_profit_pct - benchmark_profit_pct
        deal_crossings = _compute_soft_deal_crossings(predicts, valid_mask)
        deals_penalty = _compute_deals_penalty(deal_crossings, deals_lambda)
        loss = -excess_profit_pct + deals_penalty
        postfix: dict[str, str] = {
            "loss": f"{loss.item():.2f}",
            "outperf": f"{excess_profit_pct.item():.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if deals_lambda != 0.0:
            postfix["deals"] = f"{deal_crossings.item():.1f}"
            postfix["deals_pen"] = f"{deals_penalty.item():.3f}"
        progress_bar.set_postfix(**postfix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(os.environ["GRAD_MAX_NORM"]))
        grad_sq_sum = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq_sum += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = grad_sq_sum**0.5
        optimizer.step()
        scheduler.step()
        weight_sq_sum = 0.0
        for param in model.parameters():
            weight_sq_sum += float(torch.sum(param.detach() ** 2).item())
        weight_norm = weight_sq_sum**0.5
        weight_norm_change = weight_norm - prev_weight_norm
        prev_weight_norm = weight_norm
        loss_value = float(loss.item())
        final_profit_value = float(excess_profit_pct.item())
        loss_history.append(loss_value)
        profit_history.append(final_profit_value)
        grad_norm_history.append(grad_norm)
        weight_norm_history.append(weight_norm)
        weight_norm_change_history.append(weight_norm_change)
        deals_penalty_history.append(float(deals_penalty.item()))
        deal_crossings_history.append(float(deal_crossings.item()))
        strategy_profit_history.append(float(strategy_profit_pct.item()))
        benchmark_profit_history.append(float(benchmark_profit_pct.item()))

    train_info = {
        "optimizer": "AdamW",
        "lr_scheduler": "CosineAnnealingWarmRestarts",
        "device": str(device),
        "loss": "neg_excess_vs_buyhold_plus_deals_penalty",
        "learning_rate": learning_rate,
        "lr_restart_period": lr_restart_period,
        "lr_min": lr_min,
        "epochs": num_epochs,
        "epochs_completed_before_resume": start_epoch,
        "total_epochs": start_epoch + num_epochs,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "drawdown_lambda": drawdown_lambda,
        "deals_lambda": deals_lambda,
        "benchmark": "buy_and_hold_plus_one",
        "model_type": "GruPolicy",
        "gru_hidden_size": hidden_size,
        "autoregressive_prev_label": True,
        "autoregressive_prev_drawdown": True,
        "autoregressive_feature_name": AUTOREGRESSIVE_PREV_LABEL_FEATURE_NAME,
        "autoregressive_feature_names": AUTOREGRESSIVE_FEATURE_NAMES,
        "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
        "autoregressive_initial_prev_drawdown": AUTOREGRESSIVE_INITIAL_DRAWDOWN,
        "final_train_loss": loss_value,
        "final_train_profit": final_profit_value,
        "final_train_strategy_profit": strategy_profit_history[-1] if strategy_profit_history else 0.0,
        "final_benchmark_profit": benchmark_profit_history[-1] if benchmark_profit_history else 0.0,
        "final_deals_penalty": deals_penalty_history[-1] if deals_penalty_history else 0.0,
        "final_deal_crossings": deal_crossings_history[-1] if deal_crossings_history else 0.0,
        "final_grad_norm": grad_norm_history[-1] if grad_norm_history else 0.0,
        "final_weight_norm": weight_norm_history[-1] if weight_norm_history else 0.0,
    }
    train_history = {
        "loss": loss_history,
        "final_profit": profit_history,
        "strategy_profit": strategy_profit_history,
        "benchmark_profit": benchmark_profit_history,
        "grad_norm": grad_norm_history,
        "weight_norm": weight_norm_history,
        "weight_norm_change": weight_norm_change_history,
        "deals_penalty": deals_penalty_history,
        "deal_crossings": deal_crossings_history,
    }
    training_checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epochs_completed": len(loss_history),
        "prev_weight_norm": prev_weight_norm,
    }
    return model, train_info, train_history, training_checkpoint


def predict_direction(
    model: GruPolicy,
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
        predicts, _, _ = _rollout_train_timesteps(
            model,
            X_tensor,
            open_price_tensor,
            deposit_multp_tensor,
        )
        position_dirs = np.where(predicts >= score_thresholds, 1, -1).astype(np.int64)
    return position_dirs


class SelfLearn:
    def __init__(
        self,
        config: str,
        output_dir: Path,
        deposit: float = 1000.0,
        resume_checkpoint: Path | None = None,
    ):
        self.config_path = config
        self.output_dir = output_dir
        self.deposit = deposit
        self.device = resolve_device()
        self.resume_checkpoint = resume_checkpoint
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

        model, train_info, train_history, training_checkpoint = train_classifier(
            X_train,
            open_price_train,
            deposit_multp=deposit_multp_train,
            device=self.device,
            deposit=self.deposit,
            bar_description="Training: ",
            resume_checkpoint=self.resume_checkpoint,
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
            training_checkpoint=training_checkpoint,
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
        training_checkpoint = a.training_checkpoint
        model = a.model
        metrics = a.metrics
        y_pr_train = a.y_pr_train
        y_pr_test = a.y_pr_test
        train_strategy_step_profit = a.train_strategy_step_profit
        test_strategy_step_profit = a.test_strategy_step_profit

        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_feature_count = int(X_shape[2])
        in_features = base_feature_count + len(AUTOREGRESSIVE_FEATURE_NAMES)
        gru_hidden_size = int(train_info["gru_hidden_size"])
        assert training_checkpoint is not None
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": training_checkpoint["optimizer_state_dict"],
                "scheduler_state_dict": training_checkpoint["scheduler_state_dict"],
                "train_history": train_history,
                "epochs_completed": training_checkpoint["epochs_completed"],
                "prev_weight_norm": training_checkpoint["prev_weight_norm"],
                "model_type": "GruPolicy",
                "in_features": in_features,
                "base_feature_count": base_feature_count,
                "gru_hidden_size": gru_hidden_size,
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
            output_path = self.output_dir / f"strategy_profit_train_test_{symbol.ticker}.png"
            buy_hold_train = _compute_buy_hold_step_profit(
                timestamps=timestamps_train[i],
                open_price=open_price_train[i],
                deposit=self.deposit,
            )
            buy_hold_test = _compute_buy_hold_step_profit(
                timestamps=timestamps_test[i],
                open_price=open_price_test[i],
                deposit=self.deposit,
            )
            visualizer.save_strategy_train_test_profit_plot(
                timestamps_train=timestamps_train[i][:-1],
                strategy_cum_train=np.cumsum(train_strategy_step_profit[i]) if train_strategy_step_profit[i].size else np.array([], dtype=np.float64),
                timestamps_test=timestamps_test[i][:-1],
                strategy_cum_test=np.cumsum(test_strategy_step_profit[i]) if test_strategy_step_profit[i].size else np.array([], dtype=np.float64),
                pred_sign_train=y_pr_train[i][:-1],
                pred_sign_test=y_pr_test[i][:-1],
                output_path=output_path,
                buy_hold_cum_train=np.cumsum(buy_hold_train),
                buy_hold_cum_test=np.cumsum(buy_hold_test),
            )
            logger.info(f"Strategy profit plot saved to: {output_path}")

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
            "model_type": "GruPolicy",
            "model_params": {
                "in_features": in_features,
                "base_feature_count": base_feature_count,
                "gru_hidden_size": gru_hidden_size,
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
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="Continue training from checkpoint (default: ML_OUTPUT_DIR/model.pt)",
    )
    args = parser.parse_args()

    if os.environ.get("ML_OUTPUT_DIR") is None:
        logger.error("ML_OUTPUT_DIR is not set")
        return 1
    output_dir = Path(os.environ.get("ML_OUTPUT_DIR")).resolve()
    resume_checkpoint = resolve_resume_checkpoint(output_dir, cli_path=args.resume)
    self_learner = SelfLearn(
        config=os.environ.get("CONFIG"),
        output_dir=output_dir,
        deposit=float(os.environ.get("DEPOSIT")),
        resume_checkpoint=resume_checkpoint,
    )
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
