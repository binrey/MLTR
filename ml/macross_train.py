"""Train/evaluate Linear Regression baseline for macross direction labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
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


def predict_direction(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    # Map continuous regression output back to direction labels {-1, 1}.
    return np.where(model.predict(X) >= 0.0, 1, -1).astype(np.int64)


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

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pr_train = predict_direction(model, X_train)
    y_pr_test = predict_direction(model, X_test)

    metrics = {
        "train": evaluate(y_train, y_pr_train),
        "test": evaluate(y_test, y_pr_test),
        "dataset": {
            "aligned_rows": meta["aligned_rows"],
            "class_balance": meta["class_balance"],
            "train_size": train_size,
            "test_size": test_size,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
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
    schema = {
        "feature_names": meta["feature_names"],
        "model_type": "LinearRegression",
        "model_params": model.get_params(),
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> int:
    if os.environ.get("ML_OUTPUT_DIR") is None:
        logger.error("ML_OUTPUT_DIR is not set")
        return 1
    output_dir = Path(os.environ.get("ML_OUTPUT_DIR")).resolve() / "macross"
    metrics = train_and_save(output_dir)
    logger.info(f"Linear regression artifacts saved to: {output_dir}")
    logger.info(f"Train accuracy: {metrics['train']['accuracy']:.4f}")
    logger.info(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    logger.info(f"Test balanced accuracy: {metrics['test']['balanced_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
