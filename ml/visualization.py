"""Visualization utilities for macross model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class PredictionVisualizer:
    """Creates and saves prediction-vs-ground-truth plots."""
    def __init__(self, deposit: float = 1000.0, open_price_train: np.ndarray = None, open_price_test: np.ndarray = None):
        self.deposit_train = deposit / open_price_train
        self.deposit_test = deposit / open_price_test

    def save_buy_and_hold_comparison_plot(
        self,
        timestamps_train: np.ndarray,
        buy_and_hold_cum_train: np.ndarray,
        strategy_cum_train: np.ndarray,
        timestamps_test: np.ndarray,
        buy_and_hold_cum_test: np.ndarray,
        strategy_cum_test: np.ndarray,
        pred_sign_train: np.ndarray | None,
        pred_sign_test: np.ndarray | None,
        output_path: Path,
    ) -> None:
        fig, (ax_profit, ax_sign) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(14, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_profit.plot(
            timestamps_train,
            buy_and_hold_cum_train,
            label="buy_and_hold_train_cumsum",
            color="tab:blue",
            linewidth=1.3,
            alpha=0.9,
        )
        ax_profit.plot(
            timestamps_train,
            strategy_cum_train,
            label="strategy_train_cumsum",
            color="tab:orange",
            linewidth=1.8,
            alpha=0.9,
        )
        ax_profit.plot(
            timestamps_test,
            buy_and_hold_cum_test,
            label="buy_and_hold_test_cumsum",
            color="tab:green",
            linewidth=1.3,
            alpha=0.9,
        )
        ax_profit.plot(
            timestamps_test,
            strategy_cum_test,
            label="strategy_test_cumsum",
            color="tab:red",
            linewidth=1.8,
            alpha=0.9,
        )
        if timestamps_test.size > 0:
            ax_profit.axvline(timestamps_test[0], color="gray", linestyle=":", linewidth=1.0, label="train/test split")
        ax_profit.set_title("Strategy vs Buy-and-Hold Cumulative Profit")
        ax_profit.set_ylabel("Cumulative profit (price units)")
        ax_profit.grid(True, alpha=0.2)
        ax_profit.legend(loc="upper left", ncol=2)

        if pred_sign_train is not None and pred_sign_train.size > 0:
            ax_sign.step(
                timestamps_train,
                pred_sign_train,
                where="post",
                label="pred_sign_train",
                color="tab:orange",
                linewidth=1.1,
                alpha=0.9,
            )
        if pred_sign_test is not None and pred_sign_test.size > 0:
            ax_sign.step(
                timestamps_test,
                pred_sign_test,
                where="post",
                label="pred_sign_test",
                color="tab:red",
                linewidth=1.1,
                alpha=0.9,
            )
        if timestamps_test.size > 0:
            ax_sign.axvline(timestamps_test[0], color="gray", linestyle=":", linewidth=1.0)
        ax_sign.set_ylabel("Pred sign")
        ax_sign.set_yticks([-1, 1])
        ax_sign.set_ylim(-1.2, 1.2)
        ax_sign.grid(True, alpha=0.2)
        ax_sign.set_xlabel("Date")
        ax_sign.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_sign.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_sign.xaxis.get_major_locator()))
        ax_sign.legend(loc="upper left", ncol=2)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_loss_change_plot(
        self,
        loss_values: np.ndarray,
        profit_values: np.ndarray,
        output_path: Path,
    ) -> None:
        epochs = np.arange(1, loss_values.size + 1, dtype=np.int64)
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(epochs, loss_values, label="train_loss", color="tab:red", linewidth=1.4, alpha=0.95)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (-final profit)", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            profit_values,
            label="train_final_profit",
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax2.set_ylabel("Final train profit", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        ax1.set_title("Training loss/profit change by epoch")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_gradient_change_plot(
        self,
        grad_norm_values: np.ndarray,
        grad_change_values: np.ndarray,
        output_path: Path,
    ) -> None:
        epochs = np.arange(1, grad_norm_values.size + 1, dtype=np.int64)
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(epochs, grad_norm_values, label="grad_norm_l2", color="tab:purple", linewidth=1.4, alpha=0.95)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Gradient L2 norm", color="tab:purple")
        ax1.tick_params(axis="y", labelcolor="tab:purple")
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            grad_change_values,
            label="grad_norm_change",
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax2.set_ylabel("Gradient norm change (epoch to epoch)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        ax1.set_title("Training gradient change by epoch")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_weight_norm_change_plot(
        self,
        weight_norm_values: np.ndarray,
        weight_change_values: np.ndarray,
        output_path: Path,
    ) -> None:
        epochs = np.arange(1, weight_norm_values.size + 1, dtype=np.int64)
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(epochs, weight_norm_values, label="weights_norm_l2", color="tab:brown", linewidth=1.4, alpha=0.95)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weights L2 norm", color="tab:brown")
        ax1.tick_params(axis="y", labelcolor="tab:brown")
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            weight_change_values,
            label="weights_norm_change",
            color="tab:cyan",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax2.set_ylabel("Weights norm change (epoch to epoch)", color="tab:cyan")
        ax2.tick_params(axis="y", labelcolor="tab:cyan")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        ax1.set_title("Training weights norm change by epoch")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
