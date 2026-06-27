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
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import Logger
from ml.selflearn_dataset import TRAIN_FRAC, DatasetMeta, build_dataset
from ml.utils import (
    build_valid_panel_mask,
    calendar_years_from_timestamps,
    compute_buy_hold_step_profit,
    compute_step_profit_with_boundaries,
    cv_fold_stats,
    pack_yearly_training_batch,
    resolve_commission_rate_pct,
)
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


def _default_segments(n_rows: int) -> list[tuple[int, int]]:
    return [(0, int(n_rows))] if n_rows > 0 else []


def _compute_benchmark_step_pnl_parts(
    X_tensor: torch.Tensor,
    open_price_tensor: torch.Tensor,
    deposit_multp_tensor: torch.Tensor,
    loss_mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Buy-and-hold (+1 direction) PnL using the same step logic as training rollout."""
    bsz, tlen, _ = X_tensor.shape
    sequence_pnl = torch.zeros((bsz,), dtype=X_tensor.dtype, device=X_tensor.device)
    for t in range(tlen):
        valid_t = torch.isfinite(open_price_tensor[:, t]) & torch.all(
            torch.isfinite(X_tensor[:, t, :]), dim=1
        )
        if t < tlen - 1:
            valid_next = torch.isfinite(open_price_tensor[:, t + 1]) & torch.all(
                torch.isfinite(X_tensor[:, t + 1, :]), dim=1
            )
            active = valid_t & valid_next
            if loss_mask_tensor is not None:
                active = active & loss_mask_tensor[:, t] & loss_mask_tensor[:, t + 1]
            if torch.any(active):
                open_change = (
                    open_price_tensor[active, t + 1] - open_price_tensor[active, t]
                )
                step_pnl = open_change * deposit_multp_tensor[active, t]
                step_full = torch.zeros_like(sequence_pnl)
                step_full[active] = step_pnl
                sequence_pnl = sequence_pnl + step_full
    return sequence_pnl


def _rollout_train_timesteps(model: GruPolicy,
                             X_tensor: torch.Tensor,
                             open_price_tensor: torch.Tensor,
                             deposit_multp_tensor: torch.Tensor,
                             loss_mask_tensor: torch.Tensor | None = None,
                             detach_predictions: bool = True,
                             commission_rate_pct: float = 0.0,
                             ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, tlen, _ = X_tensor.shape
    device = X_tensor.device
    dtype = X_tensor.dtype
    sequence_gross_pnl = torch.zeros((bsz,), dtype=dtype, device=device)
    sequence_commission = torch.zeros((bsz,), dtype=dtype, device=device)
    predicts: torch.Tensor = torch.zeros((bsz, tlen), dtype=dtype, device=device)
    valid_mask = torch.zeros((bsz, tlen), dtype=torch.bool, device=device)
    prev_state = torch.zeros((bsz, 2), dtype=dtype, device=device)
    prev_pred = torch.zeros(bsz, dtype=dtype, device=device)
    prev_valid = torch.zeros(bsz, dtype=torch.bool, device=device)
    hidden = torch.zeros((bsz, model.hidden_size), dtype=dtype, device=device)
    for t in range(tlen):
        valid_t = torch.isfinite(open_price_tensor[:, t]) & torch.all(
            torch.isfinite(X_tensor[:, t, :]), dim=1
        )
        if not torch.any(valid_t):
            # No usable rows this timestep: reset hidden autoregressive state.
            prev_state = prev_state * 0.0
            hidden = hidden * 0.0
            prev_pred = prev_pred * 0.0
            prev_valid = prev_valid & False
            continue
        loss_t = (
            loss_mask_tensor[:, t]
            if loss_mask_tensor is not None
            else torch.ones((bsz,), dtype=torch.bool, device=device)
        )
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
        if commission_rate_pct > 0.0 and t > 0:
            prev_loss_t = (
                loss_mask_tensor[:, t - 1]
                if loss_mask_tensor is not None
                else torch.ones((bsz,), dtype=torch.bool, device=device)
            )
            transition_valid = valid_t & prev_valid & loss_t & prev_loss_t
            if torch.any(transition_valid):
                deal_delta = (
                    predicts_tensor[transition_valid[valid_t]]
                    - prev_pred[transition_valid]
                ).abs()
                step_fee = (
                    open_price_tensor[transition_valid, t - 1]
                    * deal_delta
                    * deposit_multp_tensor[transition_valid, t - 1]
                    * (commission_rate_pct / 100.0)
                )
                step_full = torch.zeros_like(sequence_commission)
                step_full[transition_valid] = step_fee
                sequence_commission = sequence_commission + step_full
        # Update autoregressive state only for active batch elements.
        next_prev_state = prev_state.clone()
        next_prev_state[valid_t, 0] = predicts_tensor
        prev_state = next_prev_state
        next_prev_pred = prev_pred.clone()
        next_prev_pred[valid_t] = predicts_tensor.detach() if detach_predictions else predicts_tensor
        prev_pred = next_prev_pred
        prev_valid = valid_t.clone()
        next_hidden = hidden.clone()
        next_hidden[valid_t] = hidden_t
        hidden = next_hidden
        if t < tlen - 1:
            valid_next = torch.isfinite(open_price_tensor[:, t + 1]) & torch.all(
                torch.isfinite(X_tensor[:, t + 1, :]), dim=1
            )
            active = valid_t & valid_next
            if loss_mask_tensor is not None:
                active = active & loss_t & loss_mask_tensor[:, t + 1]
            if torch.any(active):
                open_change = (
                    open_price_tensor[active, t + 1] - open_price_tensor[active, t]
                )
                dir_active = prev_state[active, 0]
                step_pnl = open_change * dir_active * deposit_multp_tensor[active, t]
                step_full = torch.zeros_like(sequence_gross_pnl)
                step_full[active] = step_pnl
                sequence_gross_pnl = sequence_gross_pnl + step_full
        # Reset states for currently invalid rows (dynamic batch reduction).
        prev_state = prev_state * valid_t[:, None].to(dtype)
        hidden = hidden * valid_t[:, None].to(dtype)
        prev_pred = prev_pred * valid_t.to(dtype)
        prev_valid = prev_valid & valid_t
    if detach_predictions:
        return (
            predicts.detach().cpu().numpy(),
            sequence_gross_pnl,
            sequence_commission,
            valid_mask,
        )
    return predicts, sequence_gross_pnl, sequence_commission, valid_mask


def _compute_hold_fraction(
    predicts: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence soft fraction of saturated same-direction consecutive steps."""
    pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    if loss_mask is not None:
        pair_valid = pair_valid & loss_mask[:, :-1] & loss_mask[:, 1:]
    sequence_valid = torch.any(pair_valid, dim=1)
    if not torch.any(pair_valid):
        return (
            torch.zeros((predicts.shape[0],), dtype=predicts.dtype, device=predicts.device),
            sequence_valid,
        )
    same_dir = torch.sigmoid(predicts[:, :-1] * predicts[:, 1:] * 8.0)
    sat_prev = torch.sigmoid((predicts[:, :-1].abs() - 0.3) * 8.0)
    sat_next = torch.sigmoid((predicts[:, 1:].abs() - 0.3) * 8.0)
    hold_scores = same_dir * sat_prev * sat_next
    pair_counts = pair_valid.sum(dim=1).clamp_min(1).to(predicts.dtype)
    hold_fraction = (hold_scores * pair_valid.to(predicts.dtype)).sum(dim=1) / pair_counts
    hold_fraction = torch.where(sequence_valid, hold_fraction, torch.zeros_like(hold_fraction))
    return hold_fraction, sequence_valid


def _compute_hold_fraction_penalty(
    predicts: torch.Tensor,
    valid_mask: torch.Tensor,
    hold_lambda: float,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sequence penalty for saturated buy/sell-and-hold segments."""
    hold_fraction, sequence_valid = _compute_hold_fraction(predicts, valid_mask, loss_mask)
    if hold_lambda == 0.0:
        return torch.zeros_like(hold_fraction), hold_fraction, sequence_valid
    return hold_lambda * hold_fraction, hold_fraction, sequence_valid


def train_classifier(X_train: np.ndarray,
                     open_price_train: np.ndarray, 
                     deposit_multp: np.ndarray,
                     device: torch.device,
                     deposit: float,
                     bar_description: str = "Training",
                     loss_mask: np.ndarray | None = None,
                     training_batch_info: dict | None = None,
                     segments: list[tuple[int, int]] | None = None,
                     resume_checkpoint: Path | None = None,
                     commission_rate_pct: float = 0.0,
                    ) -> tuple[GruPolicy, dict, dict, dict]:
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (B,T,F), got {X_train.shape}")
    bsz, tlen, feat_n = X_train.shape
    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price_train.astype(np.float32)).to(device)
    deposit_multp_tensor = torch.tensor(deposit_multp, dtype=torch.float32, device=device)
    loss_mask_tensor = (
        torch.from_numpy(loss_mask.astype(bool)).to(device)
        if loss_mask is not None
        else None
    )
    segments = segments if segments is not None else _default_segments(tlen)
    if not segments:
        raise ValueError("No valid segments available for training")

    hidden_size = int(os.environ.get("GRU_HIDDEN_SIZE", "16"))
    logit_clip = float(os.environ["LOGIT_CLIP"])
    learning_rate = float(os.environ["LR"])
    num_epochs = int(os.environ["EPOCHS"])
    drawdown_lambda = float(os.environ["DRAWDOWN_LAMBDA"])
    hold_lambda = float(os.environ.get("HOLD_LAMBDA", "0"))
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
    hold_penalty_history: list[float] = _history_list(checkpoint, "hold_penalty")
    hold_fraction_history: list[float] = _history_list(checkpoint, "hold_fraction")
    commission_history: list[float] = _history_list(checkpoint, "commission")
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
        loss_mask_tensor,
    )
    benchmark_profit_pct_by_sequence = benchmark_step_pnl_parts / deposit * 100
    base_valid_mask = torch.isfinite(open_price_tensor) & torch.all(
        torch.isfinite(X_tensor), dim=2
    )
    target_valid_mask = (
        base_valid_mask & loss_mask_tensor
        if loss_mask_tensor is not None
        else base_valid_mask
    )
    sequence_valid_mask = torch.any(target_valid_mask[:, :-1] & target_valid_mask[:, 1:], dim=1)
    if not torch.any(sequence_valid_mask):
        raise ValueError("No valid target-year transitions available for training loss")
    benchmark_profit_pct = benchmark_profit_pct_by_sequence[sequence_valid_mask].mean()
    if not torch.isfinite(benchmark_profit_pct):
        raise FloatingPointError(
            f"Non-finite benchmark profit before training: {benchmark_profit_pct.item()}"
        )

    progress_bar = tqdm(
        range(num_epochs),
        desc=bar_description,
        initial=start_epoch,
        total=start_epoch + num_epochs,
    )
    for _ in progress_bar:
        optimizer.zero_grad()
        predicts, sequence_gross_pnl, sequence_commission, valid_mask = _rollout_train_timesteps(
            model,
            X_tensor,
            open_price_tensor,
            deposit_multp_tensor,
            loss_mask_tensor=loss_mask_tensor,
            detach_predictions=False,
            commission_rate_pct=commission_rate_pct,
        )

        net_sequence_profit = sequence_gross_pnl - sequence_commission
        strategy_profit_pct_by_sequence = net_sequence_profit / deposit * 100
        commission_pct_by_sequence = sequence_commission / deposit * 100
        hold_penalty_by_sequence, hold_fraction_by_sequence, hold_valid_mask = _compute_hold_fraction_penalty(
            predicts,
            valid_mask,
            hold_lambda,
            loss_mask_tensor,
        )
        valid_loss_sequences = sequence_valid_mask & hold_valid_mask
        if not torch.any(valid_loss_sequences):
            raise ValueError("No valid packed yearly sequences available for loss")
        excess_profit_pct_by_sequence = (
            strategy_profit_pct_by_sequence - benchmark_profit_pct_by_sequence
        )
        sequence_loss = -excess_profit_pct_by_sequence + hold_penalty_by_sequence
        loss = sequence_loss[valid_loss_sequences].mean()
        gross_profit = sequence_gross_pnl[valid_loss_sequences].sum()
        commission_total = sequence_commission[valid_loss_sequences].sum()
        strategy_profit_pct = strategy_profit_pct_by_sequence[valid_loss_sequences].mean()
        benchmark_profit_pct = benchmark_profit_pct_by_sequence[valid_loss_sequences].mean()
        commission_pct = commission_pct_by_sequence[valid_loss_sequences].mean()
        hold_fraction = hold_fraction_by_sequence[valid_loss_sequences].mean()
        hold_penalty = hold_penalty_by_sequence[valid_loss_sequences].mean()
        excess_profit_pct = excess_profit_pct_by_sequence[valid_loss_sequences].mean()
        loss_components = {
            "loss": loss,
            "gross_profit": gross_profit,
            "commission_total": commission_total,
            "strategy_profit_pct": strategy_profit_pct,
            "benchmark_profit_pct": benchmark_profit_pct,
            "excess_profit_pct": excess_profit_pct,
            "hold_fraction": hold_fraction,
            "hold_penalty": hold_penalty,
        }
        non_finite_components = [
            f"{name}={float(value.detach().cpu().item())}"
            for name, value in loss_components.items()
            if not torch.isfinite(value)
        ]
        if non_finite_components:
            raise FloatingPointError(
                "Non-finite training value before backward: "
                + ", ".join(non_finite_components)
            )
        postfix: dict[str, str] = {
            "loss": f"{loss.item():.2f}",
            "outperf": f"{excess_profit_pct.item():.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if hold_lambda != 0.0:
            postfix["hold"] = f"{hold_fraction.item():.2f}"
            postfix["hold_pen"] = f"{hold_penalty.item():.2f}"
        if commission_rate_pct != 0.0:
            postfix["fees"] = f"{commission_pct.item():.2f}"
        progress_bar.set_postfix(**postfix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(os.environ["GRAD_MAX_NORM"]),
            error_if_nonfinite=True,
        )
        grad_sq_sum = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq_sum += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = grad_sq_sum**0.5
        optimizer.step()
        scheduler.step()
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError(
                    f"Non-finite model parameter after optimizer step: {name}"
                )
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
        hold_penalty_history.append(float(hold_penalty.item()))
        hold_fraction_history.append(float(hold_fraction.item()))
        commission_history.append(float(commission_pct.item()))
        strategy_profit_history.append(float(strategy_profit_pct.item()))
        benchmark_profit_history.append(float(benchmark_profit_pct.item()))

    train_info = {
        "optimizer": "AdamW",
        "lr_scheduler": "CosineAnnealingWarmRestarts",
        "device": str(device),
        "loss": "neg_excess_vs_buyhold_plus_hold_penalty_minus_commissions",
        "loss_reduction": (
            training_batch_info.get("loss_reduction")
            if training_batch_info is not None
            else "global_sequence_sum"
        ),
        "train_batching": (
            training_batch_info.get("train_batching")
            if training_batch_info is not None
            else "full_panel"
        ),
        "learning_rate": learning_rate,
        "lr_restart_period": lr_restart_period,
        "lr_min": lr_min,
        "epochs": num_epochs,
        "epochs_completed_before_resume": start_epoch,
        "total_epochs": start_epoch + num_epochs,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "drawdown_lambda": drawdown_lambda,
        "hold_lambda": hold_lambda,
        "commission_rate_pct": commission_rate_pct,
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
        "final_hold_penalty": hold_penalty_history[-1] if hold_penalty_history else 0.0,
        "final_hold_fraction": hold_fraction_history[-1] if hold_fraction_history else 0.0,
        "final_commission_pct": commission_history[-1] if commission_history else 0.0,
        "final_grad_norm": grad_norm_history[-1] if grad_norm_history else 0.0,
        "final_weight_norm": weight_norm_history[-1] if weight_norm_history else 0.0,
    }
    if training_batch_info is not None:
        train_info["training_batch"] = training_batch_info
    train_history = {
        "loss": loss_history,
        "final_profit": profit_history,
        "strategy_profit": strategy_profit_history,
        "benchmark_profit": benchmark_profit_history,
        "grad_norm": grad_norm_history,
        "weight_norm": weight_norm_history,
        "weight_norm_change": weight_norm_change_history,
        "hold_penalty": hold_penalty_history,
        "hold_fraction": hold_fraction_history,
        "commission": commission_history,
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
        predicts, _, _, _ = _rollout_train_timesteps(
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
        self.commission_rate_pct = resolve_commission_rate_pct(config)
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

        valid_train_panel = build_valid_panel_mask(X_train, open_price_train)
        valid_test_panel = build_valid_panel_mask(X_test, open_price_test)

        if not np.any(valid_train_panel):
            raise ValueError("No valid training rows after NaN filtering")
        deposit_multp = np.zeros_like(open_price, dtype=np.float64)
        pos_mask = np.isfinite(open_price) & (open_price > 0.0)
        deposit_multp[pos_mask] = self.deposit / open_price[pos_mask]
        deposit_multp_train = deposit_multp[:, :train_size]
        deposit_multp_test = deposit_multp[:, train_size:]
        packed_train = pack_yearly_training_batch(
            X_train=X_train,
            timestamps_train=timestamps_train,
            open_price_train=open_price_train,
            deposit_multp_train=deposit_multp_train,
        )

        model, train_info, train_history, training_checkpoint = train_classifier(
            packed_train.X,
            packed_train.open_price,
            deposit_multp=packed_train.deposit_multp,
            device=self.device,
            deposit=self.deposit,
            bar_description="Training: ",
            loss_mask=packed_train.loss_mask,
            training_batch_info=packed_train.metadata,
            resume_checkpoint=self.resume_checkpoint,
            commission_rate_pct=self.commission_rate_pct,
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

        train_strategy_step_profit = compute_step_profit_with_boundaries(
            timestamps=timestamps_train,
            open_price=open_price_train,
            direction=y_pr_train_panel,
            deposit_multp=deposit_multp_train,
            valid_rows=valid_train_panel,
            commission_rate_pct=self.commission_rate_pct,
        )
        test_strategy_step_profit = compute_step_profit_with_boundaries(
            timestamps=timestamps_test,
            open_price=open_price_test,
            direction=y_pr_test_panel,
            deposit_multp=deposit_multp_test,
            valid_rows=valid_test_panel,
            commission_rate_pct=self.commission_rate_pct,
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
                "training_batch": packed_train.metadata,
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
                    "test_strategy_profit_sum": cv_fold_stats([]),
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
            "test_strategy_profit_sum": cv_fold_stats(test_strategy_values),
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
            buy_hold_train = compute_buy_hold_step_profit(
                timestamps=timestamps_train[i],
                open_price=open_price_train[i],
                deposit=self.deposit,
            )
            buy_hold_test = compute_buy_hold_step_profit(
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
