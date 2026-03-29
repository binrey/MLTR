"""Train a one-layer PyTorch strategy model and compare with buy-and-hold."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import Logger
from ml.selflearn_dataset import TRAIN_FRAC, DatasetMeta, build_direction_dataset
from ml.visualization import PredictionVisualizer
from loguru import logger
from tqdm import tqdm

AUTOREGRESSIVE_FEATURE_NAME = "prev_pred_label"
AUTOREGRESSIVE_INITIAL_LABEL = 0.0

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
    def __init__(self, in_features: int):
        super().__init__()
        self.hidden = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        return self.out(x).squeeze(-1)

@dataclass
class TrainArtifacts:
    """Saved training outputs for `SelfLearn.save()` (single-window or multi-run)."""

    mode: Literal["single", "multi"]
    X: np.ndarray | None = None
    timestamps_train: np.ndarray | None = None
    timestamps_test: np.ndarray | None = None
    open_price_train: np.ndarray | None = None
    open_price_test: np.ndarray | None = None
    train_info: dict | None = None
    train_history: dict | None = None
    feat_mean: np.ndarray | None = None
    feat_std: np.ndarray | None = None
    model: OneLayerClassifier | None = None
    metrics: dict | None = None
    y_pr_train: np.ndarray | None = None
    y_pr_test: np.ndarray | None = None
    train_buy_and_hold_cum_profit: np.ndarray | None = None
    test_buy_and_hold_cum_profit: np.ndarray | None = None
    train_strategy_step_profit: np.ndarray | None = None
    test_strategy_step_profit: np.ndarray | None = None
    result: dict | None = None
    run_train_strategy_cum_profit: list[np.ndarray] | None = None
    run_test_strategy_cum_profit: list[np.ndarray] | None = None
    run_timestamps_train_steps: list[np.ndarray] | None = None
    run_timestamps_test_steps: list[np.ndarray] | None = None
    open_price: np.ndarray | None = None


def resolve_device() -> torch.device:
    requested = os.environ.get("DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda"}:
        logger.warning(f"Unsupported DEVICE='{requested}', falling back to 'cpu'")
        requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE='cuda' requested but CUDA is unavailable, falling back to 'cpu'")
        requested = "cpu"
    return torch.device(requested)


def rollout_autoregressive_logits(
    model: OneLayerClassifier,
    X_tensor: torch.Tensor,
    logit_clip: float,
) -> torch.Tensor:
    if X_tensor.shape[0] == 0:
        return torch.empty(0, dtype=X_tensor.dtype, device=X_tensor.device)
    logits: list[torch.Tensor] = []
    prev_state = torch.tensor(
        [AUTOREGRESSIVE_INITIAL_LABEL],
        dtype=X_tensor.dtype,
        device=X_tensor.device,
    )
    clip = max(float(logit_clip), 1e-6)
    for idx in range(X_tensor.shape[0]):
        step_input = torch.cat((X_tensor[idx], prev_state), dim=0).unsqueeze(0)
        step_logit = model(step_input)[0]
        logits.append(step_logit)
        step_logit_stable = clip * torch.tanh(step_logit / clip)
        prev_state = torch.tanh(step_logit_stable).view(1)
    return torch.stack(logits)


def train_classifier(X_train: np.ndarray, 
                     open_price_train: np.ndarray, 
                     deposit_multp: np.ndarray,
                     device: torch.device,
                     deposit: float) -> tuple[OneLayerClassifier, dict, dict]:
    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price_train.astype(np.float32)).to(device)
    open_change = torch.diff(open_price_tensor)
    deposit_multp_tensor = torch.tensor(deposit_multp, dtype=torch.float32, device=device)

    model = OneLayerClassifier(in_features=X_train.shape[1] + 1).to(device)
    learning_rate = float(os.environ["LR"])
    num_epochs = int(os.environ["EPOCHS"])
    logit_clip = float(os.environ["LOGIT_CLIP"])
    drawdown_lambda = float(os.environ["DRAWDOWN_LAMBDA"])
    l2_lambda = float(os.environ.get("L2_LAMBDA", "0"))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_value = 0.0
    final_profit_value = 0.0
    loss_history: list[float] = []
    profit_history: list[float] = []
    grad_norm_history: list[float] = []
    grad_norm_change_history: list[float] = []
    weight_norm_history: list[float] = []
    weight_norm_change_history: list[float] = []
    drawdown_penalty_history: list[float] = []
    l2_penalty_history: list[float] = []
    use_drawdown_penalty = drawdown_lambda != 0.0
    use_l2 = l2_lambda != 0.0
    prev_grad_norm = 0.0
    prev_weight_norm = 0.0
    progress_bar = tqdm(range(num_epochs), desc="Training")
    for _ in progress_bar:
        optimizer.zero_grad()
        logits = rollout_autoregressive_logits(model, X_tensor, logit_clip)
        # Soft-clamp logits to avoid hard zero gradients from torch.clamp outside bounds.
        logits_stable = logit_clip * torch.tanh(logits / max(logit_clip, 1e-6))
        # Differentiable trading direction in [-1, 1].
        direction = torch.tanh(logits_stable)
        if open_change.numel() > 0:
            # Direction at t is applied to open price change from t -> t+1.
            step_pnl = open_change * direction[:-1] * deposit_multp_tensor[:-1]
            sequence_profit = torch.sum(step_pnl)
            if use_drawdown_penalty:
                equity_curve = torch.cumsum(step_pnl, dim=0)
                running_peak = torch.cummax(equity_curve, dim=0).values
                drawdown = running_peak - equity_curve
                drawdown_penalty = torch.mean(drawdown)
            else:
                drawdown_penalty = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
        else:
            sequence_profit = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
            drawdown_penalty = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
        final_profit = sequence_profit
        buy_and_hold_profit = torch.abs(torch.sum(open_change * deposit_multp_tensor[:-1]))
        bh_denom = torch.clamp(torch.abs(buy_and_hold_profit), min=1e-6)
        relative_outperformance = final_profit / deposit #(final_profit - buy_and_hold_profit) / bh_denom
        if use_l2:
            l2_penalty = sum((p * p).sum() for p in model.parameters())
        else:
            l2_penalty = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
        loss = (
            -relative_outperformance
            + drawdown_lambda * drawdown_penalty
            + l2_lambda * l2_penalty
        )
        postfix: dict[str, str] = {
            "loss": f"{loss.item():.3f}",
            "rel_outperf": f"{relative_outperformance.item():.3f}",
            "drawdown": f"{drawdown_penalty.item():.3f}",
        }
        if use_l2:
            postfix["l2"] = f"{l2_penalty.item():.3f}"
        progress_bar.set_postfix(**postfix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        grad_sq_sum = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq_sum += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = grad_sq_sum**0.5
        grad_norm_change = grad_norm - prev_grad_norm
        prev_grad_norm = grad_norm
        optimizer.step()
        weight_sq_sum = 0.0
        for param in model.parameters():
            weight_sq_sum += float(torch.sum(param.detach() ** 2).item())
        weight_norm = weight_sq_sum**0.5
        weight_norm_change = weight_norm - prev_weight_norm
        prev_weight_norm = weight_norm
        loss_value = float(loss.item())
        final_profit_value = float(final_profit.item())
        loss_history.append(loss_value)
        profit_history.append(final_profit_value)
        grad_norm_history.append(grad_norm)
        grad_norm_change_history.append(grad_norm_change)
        weight_norm_history.append(weight_norm)
        weight_norm_change_history.append(weight_norm_change)
        drawdown_penalty_history.append(float(drawdown_penalty.item()))
        l2_penalty_history.append(float(l2_penalty.item()))

    train_info = {
        "optimizer": "Adam",
        "device": str(device),
        "loss": "neg_relative_outperformance_plus_drawdown_penalty_plus_optional_l2",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "logit_clip": logit_clip,
        "drawdown_lambda": drawdown_lambda,
        "l2_lambda": l2_lambda,
        "autoregressive_prev_label": True,
        "autoregressive_feature_name": AUTOREGRESSIVE_FEATURE_NAME,
        "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
        "final_train_loss": loss_value,
        "final_train_profit": final_profit_value,
        "final_drawdown_penalty": drawdown_penalty_history[-1] if drawdown_penalty_history else 0.0,
        "final_l2_penalty": l2_penalty_history[-1] if l2_penalty_history else 0.0,
        "final_grad_norm": grad_norm_history[-1] if grad_norm_history else 0.0,
        "final_weight_norm": weight_norm_history[-1] if weight_norm_history else 0.0,
    }
    train_history = {
        "loss": loss_history,
        "final_profit": profit_history,
        "grad_norm": grad_norm_history,
        "grad_norm_change": grad_norm_change_history,
        "weight_norm": weight_norm_history,
        "weight_norm_change": weight_norm_change_history,
        "drawdown_penalty": drawdown_penalty_history,
        "l2_penalty": l2_penalty_history,
    }
    return model, train_info, train_history


def predict_direction(model: OneLayerClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = rollout_autoregressive_logits(
            model,
            torch.from_numpy(X.astype(np.float32)).to(device),
            logit_clip=float(os.environ["LOGIT_CLIP"]),
        )
    return np.where(logits.cpu().numpy() >= 0.0, 1, -1).astype(np.int64)


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
        buy_and_hold_final_profit: float,
        train_size: int | None = None,
    ) -> TrainArtifacts:
        deposit_multp = self.deposit / open_price

        n = X.shape[0]
        if train_size is None:
            train_size = int(np.round(n * TRAIN_FRAC))
        train_size = max(1, min(train_size, n - 1))
        test_size = n - train_size

        X_train = X[:train_size]
        X_test = X[train_size:]
        timestamps_train = timestamps[:train_size]
        timestamps_test = timestamps[train_size:]
        open_price_train = open_price[:train_size]
        open_price_test = open_price[train_size:]
        deposit_multp_train = deposit_multp[:train_size]
        deposit_multp_test = deposit_multp[train_size:]

        feat_mean = X_train.mean(axis=0, dtype=np.float64)
        feat_std = X_train.std(axis=0, dtype=np.float64)
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
        X_train_scaled = ((X_train - feat_mean) / feat_std).astype(np.float32)
        X_test_scaled = ((X_test - feat_mean) / feat_std).astype(np.float32)
        X_scaled = ((X - feat_mean) / feat_std).astype(np.float32)

        model, train_info, train_history = train_classifier(
            X_train_scaled,
            open_price_train,
            deposit_multp=deposit_multp_train,
            device=self.device,
            deposit=self.deposit,
        )
        y_pr_train = predict_direction(model, X_train_scaled, device=self.device)
        y_pr_test = predict_direction(model, X_test_scaled, device=self.device)
        y_pr_all = predict_direction(model, X_scaled, device=self.device)
        train_buy_and_hold_step_profit = np.diff(open_price_train) * deposit_multp_train[:-1]
        test_buy_and_hold_step_profit = np.diff(open_price_test) * deposit_multp_test[:-1]
        train_buy_and_hold_cum_profit = (
            np.cumsum(train_buy_and_hold_step_profit)
            if train_buy_and_hold_step_profit.size
            else np.array([], dtype=np.float64)
        )
        test_buy_and_hold_cum_profit = (
            np.cumsum(test_buy_and_hold_step_profit)
            + (float(train_buy_and_hold_cum_profit[-1]) if train_buy_and_hold_cum_profit.size else 0.0)
            if test_buy_and_hold_step_profit.size
            else np.array([], dtype=np.float64)
        )
        train_strategy_step_profit = np.diff(open_price_train) * y_pr_train[:-1] * deposit_multp_train[:-1]
        test_strategy_step_profit = np.diff(open_price_test) * y_pr_test[:-1] * deposit_multp_test[:-1]
        strategy_step_profit = np.diff(open_price) * y_pr_all[:-1]
        strategy_cum_profit = np.cumsum(strategy_step_profit) if strategy_step_profit.size else np.array([], dtype=np.float64)
        strategy_final_profit = float(strategy_cum_profit[-1]) if strategy_cum_profit.size else 0.0

        metrics = {
            "dataset": {
                "aligned_rows": int(n),
                "train_size": train_size,
                "test_size": test_size,
            },
            "training": train_info,
            "profit": {
                "strategy_final_profit": strategy_final_profit,
                "buy_and_hold_final_profit": buy_and_hold_final_profit,
                "relative_outperformance_vs_buy_and_hold": float(
                    (strategy_final_profit - buy_and_hold_final_profit) / max(abs(buy_and_hold_final_profit), 1e-6)
                ),
            },
        }

        return TrainArtifacts(
            mode="single",
            X=X,
            timestamps_train=timestamps_train,
            timestamps_test=timestamps_test,
            open_price_train=open_price_train,
            open_price_test=open_price_test,
            train_info=train_info,
            train_history=train_history,
            feat_mean=feat_mean,
            feat_std=feat_std,
            model=model,
            metrics=metrics,
            y_pr_train=y_pr_train,
            y_pr_test=y_pr_test,
            train_buy_and_hold_cum_profit=train_buy_and_hold_cum_profit,
            test_buy_and_hold_cum_profit=test_buy_and_hold_cum_profit,
            train_strategy_step_profit=train_strategy_step_profit,
            test_strategy_step_profit=test_strategy_step_profit,
        )

    def train(self) -> dict:
        X, meta = build_direction_dataset(self.config_path)
        timestamps = np.asarray(meta.timestamps)
        open_price = np.asarray(meta.open_price, dtype=np.float64)
        self._meta = meta
        self._artifacts = self._train_from_window(
            X=X,
            timestamps=timestamps,
            open_price=open_price,
            buy_and_hold_final_profit=float(meta.buy_and_hold_final_profit),
        )
        assert self._artifacts.metrics is not None
        return self._artifacts.metrics

    def train_multip(self, runs: int = 5, max_cut_frac: float = 0.1, seed: int | None = None) -> dict:
        if runs < 1:
            raise ValueError("runs must be >= 1")
        if not 0.0 <= max_cut_frac < 0.5:
            raise ValueError("max_cut_frac must be in [0.0, 0.5)")

        X, meta = build_direction_dataset(self.config_path)
        self._meta = meta
        timestamps = np.asarray(meta.timestamps)
        open_price = np.asarray(meta.open_price, dtype=np.float64)
        n = X.shape[0]
        if n < 4:
            raise ValueError("Not enough rows for train_multip; need at least 4 aligned rows")

        full_train_size = int(np.round(n * TRAIN_FRAC))
        full_train_size = max(1, min(full_train_size, n - 1))
        fixed_test_start_idx = full_train_size
        train_pool_len = full_train_size

        rng = np.random.default_rng(seed)
        max_cut = int(np.floor(train_pool_len * max_cut_frac))
        min_window_rows = 4

        run_metrics: list[dict] = []
        strategy_profit_values: list[float] = []
        rel_outperf_values: list[float] = []
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

            X_train_seg = X[train_global_start:train_global_end]
            X_test_seg = X[fixed_test_start_idx:n]
            X_window = np.concatenate([X_train_seg, X_test_seg], axis=0)
            timestamps_window = np.concatenate(
                [timestamps[train_global_start:train_global_end], timestamps[fixed_test_start_idx:n]],
                axis=0,
            )
            open_price_window = np.concatenate(
                [open_price[train_global_start:train_global_end], open_price[fixed_test_start_idx:n]],
                axis=0,
            )
            train_rows = int(X_train_seg.shape[0])
            buy_and_hold_window_step_profit = np.diff(open_price_window) * (self.deposit / open_price_window)[:-1]
            buy_and_hold_window_final_profit = (
                float(np.cumsum(buy_and_hold_window_step_profit)[-1])
                if buy_and_hold_window_step_profit.size
                else 0.0
            )

            artifacts = self._train_from_window(
                X=X_window,
                timestamps=timestamps_window,
                open_price=open_price_window,
                buy_and_hold_final_profit=buy_and_hold_window_final_profit,
                train_size=train_rows,
            )
            metrics = artifacts.metrics
            assert metrics is not None
            strategy_profit = float(metrics["profit"]["strategy_final_profit"])
            rel_outperf = float(metrics["profit"]["relative_outperformance_vs_buy_and_hold"])
            final_loss = float(metrics["training"]["final_train_loss"])
            strategy_profit_values.append(strategy_profit)
            rel_outperf_values.append(rel_outperf)
            final_loss_values.append(final_loss)
            tr_step = artifacts.train_strategy_step_profit
            te_step = artifacts.test_strategy_step_profit
            ts_tr = artifacts.timestamps_train
            ts_te = artifacts.timestamps_test
            train_sum = float(np.sum(tr_step)) if tr_step.size else 0.0
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
        rel_outperf_arr = np.asarray(rel_outperf_values, dtype=np.float64)
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
            "base_aligned_rows": n,
            "fixed_test_start_idx": int(fixed_test_start_idx),
            "fixed_test_rows": int(n - fixed_test_start_idx),
            "train_pool_rows": int(train_pool_len),
            "strategy_final_profit": _stats(strategy_profit_arr),
            "relative_outperformance_vs_buy_and_hold": _stats(rel_outperf_arr),
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
            open_price=open_price,
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

    def _save_single(self) -> dict:
        a = self._artifacts
        meta = self._meta
        assert a is not None and meta is not None
        X = a.X
        timestamps_train = a.timestamps_train
        timestamps_test = a.timestamps_test
        open_price_train = a.open_price_train
        open_price_test = a.open_price_test
        train_info = a.train_info
        train_history = a.train_history
        feat_mean = a.feat_mean
        feat_std = a.feat_std
        model = a.model
        metrics = a.metrics
        y_pr_train = a.y_pr_train
        y_pr_test = a.y_pr_test
        train_buy_and_hold_cum_profit = a.train_buy_and_hold_cum_profit
        test_buy_and_hold_cum_profit = a.test_buy_and_hold_cum_profit
        train_strategy_step_profit = a.train_strategy_step_profit
        test_strategy_step_profit = a.test_strategy_step_profit

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_features": int(X.shape[1] + 1),
                "base_feature_count": int(X.shape[1]),
                "feature_mean": feat_mean.tolist(),
                "feature_std": feat_std.tolist(),
                "autoregressive_prev_label": True,
                "autoregressive_feature_name": AUTOREGRESSIVE_FEATURE_NAME,
                "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
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
        visualizer.save_buy_and_hold_comparison_plot(
            timestamps_train=timestamps_train[:-1],
            buy_and_hold_cum_train=train_buy_and_hold_cum_profit,
            strategy_cum_train=np.cumsum(train_strategy_step_profit) if train_strategy_step_profit.size else np.array([], dtype=np.float64),
            timestamps_test=timestamps_test[:-1],
            buy_and_hold_cum_test=test_buy_and_hold_cum_profit,
            strategy_cum_test=(
                np.cumsum(test_strategy_step_profit) + float(np.sum(train_strategy_step_profit))
                if test_strategy_step_profit.size
                else np.array([], dtype=np.float64)
            ),
            pred_sign_train=y_pr_train[:-1],
            pred_sign_test=y_pr_test[:-1],
            output_path=self.output_dir / "profit_vs_buy_and_hold_train_test.png",
        )
        logger.info(f"Profit-vs-buy-and-hold plot saved to: {self.output_dir / 'profit_vs_buy_and_hold_train_test.png'}")
        visualizer.save_loss_change_plot(
            loss_values=np.asarray(train_history["loss"], dtype=np.float64),
            profit_values=np.asarray(train_history["final_profit"], dtype=np.float64),
            output_path=self.output_dir / "loss_change_train.png",
        )
        logger.info(f"Loss change plot saved to: {self.output_dir / 'loss_change_train.png'}")
        visualizer.save_gradient_change_plot(
            grad_norm_values=np.asarray(train_history["grad_norm"], dtype=np.float64),
            grad_change_values=np.asarray(train_history["grad_norm_change"], dtype=np.float64),
            output_path=self.output_dir / "gradient_change_train.png",
        )
        logger.info(f"Gradient change plot saved to: {self.output_dir / 'gradient_change_train.png'}")
        visualizer.save_weight_norm_change_plot(
            weight_norm_values=np.asarray(train_history["weight_norm"], dtype=np.float64),
            weight_change_values=np.asarray(train_history["weight_norm_change"], dtype=np.float64),
            output_path=self.output_dir / "weights_norm_change_train.png",
        )
        logger.info(f"Weights norm change plot saved to: {self.output_dir / 'weights_norm_change_train.png'}")
        schema = {
            "feature_names": meta.feature_names + [AUTOREGRESSIVE_FEATURE_NAME],
            "model_type": "OneLayerTorchBinaryClassifier",
            "model_params": {
                "in_features": int(X.shape[1] + 1),
                "base_feature_count": int(X.shape[1]),
                "out_features": 1,
                "threshold": 0.5,
                "feature_mean": feat_mean.tolist(),
                "feature_std": feat_std.tolist(),
                "autoregressive_prev_label": True,
                "autoregressive_feature_name": AUTOREGRESSIVE_FEATURE_NAME,
                "autoregressive_initial_prev_label": AUTOREGRESSIVE_INITIAL_LABEL,
                "train_info": train_info,
            },
        }
        (self.output_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        return metrics


def main() -> int:
    if os.environ.get("ML_OUTPUT_DIR") is None:
        logger.error("ML_OUTPUT_DIR is not set")
        return 1
    output_dir = Path(os.environ.get("ML_OUTPUT_DIR")).resolve()
    self_learner = SelfLearn(config=os.environ.get("CONFIG"), output_dir=output_dir, deposit=float(os.environ.get("DEPOSIT")))
    metrics = self_learner.train_multip()
    metrics = self_learner.save()
    logger.info(f"One-layer PyTorch classifier artifacts saved to: {output_dir}")
    if "profit" in metrics:
        logger.info(f"Strategy final profit: {metrics['profit']['strategy_final_profit']:.6f}")
        logger.info(f"Buy-and-hold final profit: {metrics['profit']['buy_and_hold_final_profit']:.6f}")
        logger.info(f"Relative outperformance: {metrics['profit']['relative_outperformance_vs_buy_and_hold']:.6f}")
    elif "stability" in metrics:
        logger.info(
            "Multi-run strategy final profit mean/std: "
            f"{metrics['stability']['strategy_final_profit']['mean']:.6f}/"
            f"{metrics['stability']['strategy_final_profit']['std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
