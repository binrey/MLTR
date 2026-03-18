"""Train a one-layer PyTorch strategy model and compare with buy-and-hold."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import Logger
from ml.selflearn_dataset import TRAIN_FRAC, build_direction_dataset
from ml.visualization import PredictionVisualizer
from loguru import logger
from tqdm import tqdm

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
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def resolve_device() -> torch.device:
    requested = os.environ.get("DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda"}:
        logger.warning(f"Unsupported DEVICE='{requested}', falling back to 'cpu'")
        requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE='cuda' requested but CUDA is unavailable, falling back to 'cpu'")
        requested = "cpu"
    return torch.device(requested)


def train_classifier(X_train: np.ndarray, 
                     open_price_train: np.ndarray, 
                     deposit_multp: np.ndarray,
                     device: torch.device) -> tuple[OneLayerClassifier, dict, dict]:
    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    open_price_tensor = torch.from_numpy(open_price_train.astype(np.float32)).to(device)
    open_change = torch.diff(open_price_tensor)
    deposit_multp_tensor = torch.tensor(deposit_multp, dtype=torch.float32, device=device)

    model = OneLayerClassifier(in_features=X_train.shape[1]).to(device)
    learning_rate = float(os.environ["LR"])
    num_epochs = int(os.environ["EPOCHS"])
    logit_clip = float(os.environ["LOGIT_CLIP"])
    drawdown_lambda = float(os.environ["DRAWDOWN_LAMBDA"])
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
    use_drawdown_penalty = drawdown_lambda != 0.0
    prev_grad_norm = 0.0
    prev_weight_norm = 0.0
    for _ in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        logits = model(X_tensor)
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
        buy_and_hold_profit = torch.sum(open_change * deposit_multp_tensor[:-1])
        bh_denom = torch.clamp(torch.abs(buy_and_hold_profit), min=1e-6)
        relative_outperformance = (final_profit - buy_and_hold_profit) / bh_denom
        loss = -relative_outperformance + drawdown_lambda * drawdown_penalty
        loss.backward()
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

    train_info = {
        "optimizer": "Adam",
        "device": str(device),
        "loss": "neg_relative_outperformance_plus_drawdown_penalty",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "logit_clip": logit_clip,
        "drawdown_lambda": drawdown_lambda,
        "final_train_loss": loss_value,
        "final_train_profit": final_profit_value,
        "final_drawdown_penalty": drawdown_penalty_history[-1] if drawdown_penalty_history else 0.0,
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
    }
    return model, train_info, train_history


def predict_direction(model: OneLayerClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.where(probs >= 0.5, 1, -1).astype(np.int64)


def train_and_save(output_dir: Path, deposit: float = 1000.0) -> dict:
    device = resolve_device()
    X, meta = build_direction_dataset("configs/macross/BTCUSDT.py")
    timestamps = np.asarray(meta["timestamps"])
    open_price = np.asarray(meta["open_price"], dtype=np.float64)
    deposit_multp = deposit / open_price

    n = X.shape[0]
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

    # Standardize features using train statistics to prevent logit saturation.
    feat_mean = X_train.mean(axis=0, dtype=np.float64)
    feat_std = X_train.std(axis=0, dtype=np.float64)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    X_train_scaled = ((X_train - feat_mean) / feat_std).astype(np.float32)
    X_test_scaled = ((X_test - feat_mean) / feat_std).astype(np.float32)
    X_scaled = ((X - feat_mean) / feat_std).astype(np.float32)

    model, train_info, train_history = train_classifier(X_train_scaled, open_price_train, deposit_multp=deposit_multp_train, device=device)
    y_pr_train = predict_direction(model, X_train_scaled, device=device)
    y_pr_test = predict_direction(model, X_test_scaled, device=device)
    y_pr_all = predict_direction(model, X_scaled, device=device)
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
    buy_and_hold_final_profit = float(meta["buy_and_hold_final_profit"])

    metrics = {
        "dataset": {
            "aligned_rows": meta["aligned_rows"],
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

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_features": int(X.shape[1]),
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "train_info": train_info,
        },
        output_dir / "model.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    visualizer = PredictionVisualizer(deposit=deposit, open_price_train=open_price_train, open_price_test=open_price_test)
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
        output_path=output_dir / "profit_vs_buy_and_hold_train_test.png",
    )
    logger.info(f"Profit-vs-buy-and-hold plot saved to: {output_dir / 'profit_vs_buy_and_hold_train_test.png'}")
    visualizer.save_loss_change_plot(
        loss_values=np.asarray(train_history["loss"], dtype=np.float64),
        profit_values=np.asarray(train_history["final_profit"], dtype=np.float64),
        output_path=output_dir / "loss_change_train.png",
    )
    logger.info(f"Loss change plot saved to: {output_dir / 'loss_change_train.png'}")
    visualizer.save_gradient_change_plot(
        grad_norm_values=np.asarray(train_history["grad_norm"], dtype=np.float64),
        grad_change_values=np.asarray(train_history["grad_norm_change"], dtype=np.float64),
        output_path=output_dir / "gradient_change_train.png",
    )
    logger.info(f"Gradient change plot saved to: {output_dir / 'gradient_change_train.png'}")
    visualizer.save_weight_norm_change_plot(
        weight_norm_values=np.asarray(train_history["weight_norm"], dtype=np.float64),
        weight_change_values=np.asarray(train_history["weight_norm_change"], dtype=np.float64),
        output_path=output_dir / "weights_norm_change_train.png",
    )
    logger.info(f"Weights norm change plot saved to: {output_dir / 'weights_norm_change_train.png'}")
    schema = {
        "feature_names": meta["feature_names"],
        "model_type": "OneLayerTorchBinaryClassifier",
        "model_params": {
            "in_features": int(X.shape[1]),
            "out_features": 1,
            "threshold": 0.5,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "train_info": train_info,
        },
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> int:
    if os.environ.get("ML_OUTPUT_DIR") is None:
        logger.error("ML_OUTPUT_DIR is not set")
        return 1
    output_dir = Path(os.environ.get("ML_OUTPUT_DIR")).resolve() / "macross"
    metrics = train_and_save(output_dir, deposit=float(os.environ.get("DEPOSIT", "1000.0")))
    logger.info(f"One-layer PyTorch classifier artifacts saved to: {output_dir}")
    logger.info(f"Strategy final profit: {metrics['profit']['strategy_final_profit']:.6f}")
    logger.info(f"Buy-and-hold final profit: {metrics['profit']['buy_and_hold_final_profit']:.6f}")
    logger.info(f"Relative outperformance: {metrics['profit']['relative_outperformance_vs_buy_and_hold']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
