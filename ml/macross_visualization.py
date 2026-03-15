"""Visualization utilities for macross model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class MacrossPredictionVisualizer:
    """Creates and saves prediction-vs-ground-truth plots."""

    def save_predictions_plot(
        self,
        timestamps_train: np.ndarray,
        y_gt_train: np.ndarray,
        y_pr_train: np.ndarray,
        timestamps_test: np.ndarray,
        y_gt_test: np.ndarray,
        y_pr_test: np.ndarray,
        output_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(timestamps_train, y_gt_train, label="y_gt_train", color="tab:blue", linewidth=1.0, alpha=0.9)
        ax.plot(
            timestamps_train,
            y_pr_train,
            label="y_pr_train",
            color="tab:orange",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
        ax.plot(timestamps_test, y_gt_test, label="y_gt_test", color="tab:green", linewidth=1.0, alpha=0.9)
        ax.plot(
            timestamps_test,
            y_pr_test,
            label="y_pr_test",
            color="tab:red",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
        if timestamps_test.size > 0:
            ax.axvline(timestamps_test[0], color="gray", linestyle=":", linewidth=1.0, label="train/test split")
        ax.set_title("Ground Truth vs Prediction (Train + Test)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Direction label")
        ax.set_yticks([-1, 1])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_profit_plot(
        self,
        timestamps_train: np.ndarray,
        open_change_train: np.ndarray,
        y_gt_train: np.ndarray,
        y_pr_train: np.ndarray,
        timestamps_test: np.ndarray,
        open_change_test: np.ndarray,
        y_gt_test: np.ndarray,
        y_pr_test: np.ndarray,
        output_path: Path,
    ) -> None:
        gt_train_profit = np.cumsum(open_change_train * y_gt_train)
        pr_train_profit = np.cumsum(open_change_train * y_pr_train)
        gt_train_end = gt_train_profit[-1] if gt_train_profit.size else 0.0
        pr_train_end = pr_train_profit[-1] if pr_train_profit.size else 0.0

        gt_test_profit = np.cumsum(open_change_test * y_gt_test) + gt_train_end
        pr_test_profit = np.cumsum(open_change_test * y_pr_test) + pr_train_end

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(timestamps_train, gt_train_profit, label="profit_gt_train_cumsum", color="tab:blue", linewidth=1.2, alpha=0.9)
        ax.plot(
            timestamps_train,
            pr_train_profit,
            label="profit_pr_train_cumsum",
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax.plot(timestamps_test, gt_test_profit, label="profit_gt_test_cumsum", color="tab:green", linewidth=1.2, alpha=0.9)
        ax.plot(
            timestamps_test,
            pr_test_profit,
            label="profit_pr_test_cumsum",
            color="tab:red",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        if timestamps_test.size > 0:
            ax.axvline(timestamps_test[0], color="gray", linestyle=":", linewidth=1.0, label="train/test split")
        ax.set_title("Strategy Profit Curves (Open delta * Label, cumulative)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative profit (price units)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
