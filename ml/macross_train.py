"""Train/evaluate a one-layer PyTorch classifier for macross direction labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import Logger
from ml.macross_dataset import TRAIN_FRAC, build_direction_dataset
from ml.macross_visualization import MacrossPredictionVisualizer
from loguru import logger

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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [-1, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "directional_hit_rate": float((y_true == y_pred).mean()),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def train_classifier(X_train: np.ndarray, open_price_train: np.ndarray) -> tuple[OneLayerClassifier, dict, dict]:
    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    open_change = np.diff(open_price_train.astype(np.float32))
    open_change_scale = float(np.std(open_change)) if open_change.size else 1.0
    open_change_scale = max(open_change_scale, 1e-6)
    open_change_norm = open_change / open_change_scale
    open_change_tensor = torch.from_numpy(open_change_norm)

    model = OneLayerClassifier(in_features=X_train.shape[1])
    learning_rate = float(os.environ.get("MACROSS_TORCH_LR", "0.01"))
    num_epochs = int(os.environ.get("MACROSS_TORCH_EPOCHS", "5000"))
    logit_clip = float(os.environ.get("MACROSS_LOGIT_CLIP", "6.0"))
    drawdown_lambda = float(os.environ.get("MACROSS_DRAWDOWN_LAMBDA", "0.001"))
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
    prev_grad_norm = 0.0
    prev_weight_norm = 0.0
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        # Soft-clamp logits to avoid hard zero gradients from torch.clamp outside bounds.
        logits_stable = logit_clip * torch.tanh(logits / max(logit_clip, 1e-6))
        # Differentiable trading direction in [-1, 1].
        direction = torch.tanh(logits_stable)
        if open_change_tensor.numel() > 0:
            # Direction at t is applied to open price change from t -> t+1.
            step_pnl = open_change_tensor * direction[:-1]
            sequence_profit = torch.sum(step_pnl)
            equity_curve = torch.cumsum(step_pnl, dim=0)
            running_peak = torch.cummax(equity_curve, dim=0).values
            drawdown = running_peak - equity_curve
            drawdown_penalty = torch.mean(drawdown)
        else:
            sequence_profit = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
            drawdown_penalty = torch.tensor(0.0, dtype=X_tensor.dtype, device=X_tensor.device)
        final_profit = sequence_profit
        buy_and_hold_profit = torch.sum(open_change_tensor)
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
        "loss": "neg_relative_outperformance_plus_drawdown_penalty",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "open_change_scale": open_change_scale,
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


def predict_direction(model: OneLayerClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.where(probs >= 0.5, 1, -1).astype(np.int64)


def train_and_save(output_dir: Path) -> dict:
    X, y, meta = build_direction_dataset()
    timestamps = np.asarray(meta["timestamps"])
    open_price = np.asarray(meta["open_price"], dtype=np.float64)

    n = X.shape[0]
    train_size = int(np.round(n * TRAIN_FRAC))
    train_size = max(1, min(train_size, n - 1))
    test_size = n - train_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    timestamps_train = timestamps[:train_size]
    timestamps_test = timestamps[train_size:]
    open_price_train = open_price[:train_size]
    open_price_test = open_price[train_size:]

    # Standardize features using train statistics to prevent logit saturation.
    feat_mean = X_train.mean(axis=0, dtype=np.float64)
    feat_std = X_train.std(axis=0, dtype=np.float64)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    X_train_scaled = ((X_train - feat_mean) / feat_std).astype(np.float32)
    X_test_scaled = ((X_test - feat_mean) / feat_std).astype(np.float32)

    model, train_info, train_history = train_classifier(X_train_scaled, open_price_train)
    y_pr_train = predict_direction(model, X_train_scaled)
    y_pr_test = predict_direction(model, X_test_scaled)

    metrics = {
        "train": evaluate(y_train, y_pr_train),
        "test": evaluate(y_test, y_pr_test),
        "dataset": {
            "aligned_rows": meta["aligned_rows"],
            "class_balance": meta["class_balance"],
            "train_size": train_size,
            "test_size": test_size,
        },
        "training": train_info,
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
    visualizer = MacrossPredictionVisualizer()
    visualizer.save_predictions_plot(
        timestamps_train=timestamps_train,
        y_gt_train=y_train,
        y_pr_train=y_pr_train,
        timestamps_test=timestamps_test,
        y_gt_test=y_test,
        y_pr_test=y_pr_test,
        output_path=output_dir / "predictions_train_test.png",
    )
    logger.info(f"Predictions plot saved to: {output_dir / 'predictions_train_test.png'}")
    visualizer.save_profit_plot(
        timestamps_train=timestamps_train[:-1],
        open_change_train=np.diff(open_price_train),
        y_gt_train=y_train[:-1],
        y_pr_train=y_pr_train[:-1],
        timestamps_test=timestamps_test[:-1],
        open_change_test=np.diff(open_price_test),
        y_gt_test=y_test[:-1],
        y_pr_test=y_pr_test[:-1],
        output_path=output_dir / "profit_train_test.png",
    )
    logger.info(f"Profit plot saved to: {output_dir / 'profit_train_test.png'}")
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
    metrics = train_and_save(output_dir)
    logger.info(f"One-layer PyTorch classifier artifacts saved to: {output_dir}")
    logger.info(f"Train accuracy: {metrics['train']['accuracy']:.4f}")
    logger.info(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    logger.info(f"Test balanced accuracy: {metrics['test']['balanced_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
