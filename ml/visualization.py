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
    def __init__(self, 
                 deposit: float = 1000.0, 
                 open_price_train: np.ndarray = None, 
                 open_price_test: np.ndarray = None
                 ) -> None:
        self.deposit_train = deposit / open_price_train if open_price_train is not None else None
        self.deposit_test = deposit / open_price_test if open_price_test is not None else None

    @staticmethod
    def _plot_series(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        label: str | None = None,
        color: str | None = None,
        linewidth: float = 1.0,
        alpha: float = 1.0,
        linestyle: str = "-",
    ) -> matplotlib.lines.Line2D:
        return ax.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )

    @staticmethod
    def _style_profit_axis(ax: plt.Axes, title: str) -> None:
        ax.set_title(title)
        ax.set_ylabel("Cumulative profit (price units)")
        ax.grid(True, alpha=0.2)

    @staticmethod
    def _draw_split_marker(ax: plt.Axes, split_x, with_label: bool = True) -> None:
        ax.axvline(
            split_x,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="train/test split" if with_label else None,
        )

    def save_strategy_train_test_profit_plot(
        self,
        timestamps_train: np.ndarray,
        strategy_cum_train: np.ndarray,
        timestamps_test: np.ndarray,
        strategy_cum_test: np.ndarray,
        pred_sign_train: np.ndarray | None,
        pred_sign_test: np.ndarray | None,
        output_path: Path,
        buy_hold_cum_train: np.ndarray | None = None,
        buy_hold_cum_test: np.ndarray | None = None,
    ) -> None:
        fig, (ax_profit, ax_sign) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(14, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        self._plot_series(
            ax_profit,
            timestamps_train,
            strategy_cum_train,
            label="strategy_train_cumsum",
            color="tab:orange",
            linewidth=1.8,
        )
        self._plot_series(
            ax_profit,
            timestamps_test,
            strategy_cum_test,
            label="strategy_test_cumsum",
            color="tab:red",
            linewidth=1.8,
        )
        if buy_hold_cum_train is not None and buy_hold_cum_train.size > 0:
            self._plot_series(
                ax_profit,
                timestamps_train,
                buy_hold_cum_train,
                label="buy_hold_train_cumsum",
                color="tab:blue",
                linewidth=1.4,
                linestyle="--",
            )
        if buy_hold_cum_test is not None and buy_hold_cum_test.size > 0:
            self._plot_series(
                ax_profit,
                timestamps_test,
                buy_hold_cum_test,
                label="buy_hold_test_cumsum",
                color="tab:green",
                linewidth=1.4,
                linestyle="--",
            )
        if timestamps_test.size > 0:
            self._draw_split_marker(ax_profit, timestamps_test[0], with_label=True)
        self._style_profit_axis(ax_profit, "Strategy cumulative profit (train / test)")
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
            self._draw_split_marker(ax_sign, timestamps_test[0], with_label=False)
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
        self._plot_series(ax1, epochs, loss_values, label="train_loss", color="tab:red", linewidth=1.4, alpha=0.95)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (-final profit)", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        self._plot_series(
            ax2,
            epochs,
            profit_values,
            label="train_final_profit",
            color="tab:blue",
            linestyle="--",
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
        output_path: Path,
    ) -> None:
        epochs = np.arange(1, grad_norm_values.size + 1, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(14, 5))
        self._plot_series(ax, epochs, grad_norm_values, label="grad_norm_l2", color="tab:purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 norm", color="tab:purple")
        ax.tick_params(axis="y", labelcolor="tab:purple")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        ax.set_title("Training gradient L2 norm by epoch")
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
        self._plot_series(ax1, epochs, weight_norm_values, label="weights_norm_l2", color="tab:brown")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weights L2 norm", color="tab:brown")
        ax1.tick_params(axis="y", labelcolor="tab:brown")
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        self._plot_series(
            ax2,
            epochs,
            weight_change_values,
            label="weights_norm_change",
            color="tab:cyan",
            linestyle="--",
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

    def save_multi_run_profit_plot(
        self,
        run_train_strategy_cum_profit: list[np.ndarray],
        run_test_strategy_cum_profit: list[np.ndarray],
        run_timestamps_train_steps: list[np.ndarray],
        run_timestamps_test_steps: list[np.ndarray],
        output_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 5))
        has_train = False
        has_test = False
        split_labeled = False
        n_runs = len(run_train_strategy_cum_profit)
        if not (
            n_runs == len(run_test_strategy_cum_profit)
            == len(run_timestamps_train_steps)
            == len(run_timestamps_test_steps)
        ):
            raise ValueError("Per-run train/test profit and timestamp lists must have equal length")
        for run_idx in range(n_runs):
            x_tr = np.asarray(run_timestamps_train_steps[run_idx])
            y_tr = np.asarray(run_train_strategy_cum_profit[run_idx])
            x_te = np.asarray(run_timestamps_test_steps[run_idx])
            y_te = np.asarray(run_test_strategy_cum_profit[run_idx])
            if x_tr.size != y_tr.size or x_te.size != y_te.size:
                raise ValueError(
                    f"run {run_idx}: timestamp length must match cumulative curve "
                    f"(train {x_tr.size} vs {y_tr.size}, test {x_te.size} vs {y_te.size})"
                )
            if y_tr.size == 0 and y_te.size == 0:
                continue
            if y_tr.size:
                self._plot_series(
                    ax,
                    x_tr,
                    y_tr,
                    alpha=0.35,
                    color="tab:orange",
                    label="strategy_train_cumsum (runs)" if not has_train else None,
                )
                has_train = True
            if y_te.size:
                self._plot_series(
                    ax,
                    x_te,
                    y_te,
                    alpha=0.35,
                    color="tab:red",
                    label="strategy_test_cumsum (runs)" if not has_test else None,
                )
                has_test = True
            if y_tr.size and y_te.size:
                self._draw_split_marker(
                    ax,
                    x_te.flat[0],
                    with_label=not split_labeled,
                )
                split_labeled = True
        self._style_profit_axis(ax, "Strategy Cumulative Profit Across Multiple Runs")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        if has_train or has_test:
            ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_cv_chained_oos_profit_plot(
        self,
        fold_timestamps_test_steps: list[np.ndarray],
        fold_strategy_step_profit: list[np.ndarray],
        output_path: Path,
    ) -> None:
        """
        One cumulative curve: each fold's OOS test window is placed on its true timeline;
        cumulative profit chains across folds (end of year N carries into year N+1).
        """
        n = len(fold_timestamps_test_steps)
        if n != len(fold_strategy_step_profit):
            raise ValueError("Per-fold test timestamp and step-profit lists must have equal length")

        order: list[int] = []
        for i in range(n):
            ts = np.asarray(fold_timestamps_test_steps[i])
            if ts.size == 0:
                continue
            order.append(i)
        order.sort(key=lambda i: np.min(np.asarray(fold_timestamps_test_steps[i])))

        offset_strat = 0.0
        ts_parts: list[np.ndarray] = []
        strat_parts: list[np.ndarray] = []

        for i in order:
            ts = np.asarray(fold_timestamps_test_steps[i])
            s_step = np.asarray(fold_strategy_step_profit[i], dtype=np.float64)
            if s_step.size == 0 or ts.size != s_step.size:
                continue
            cum_s = offset_strat + np.cumsum(s_step)
            ts_parts.append(ts)
            strat_parts.append(cum_s)
            offset_strat = float(cum_s[-1])

        fig, ax = plt.subplots(figsize=(14, 5))
        if ts_parts:
            full_ts = np.concatenate(ts_parts)
            full_strat = np.concatenate(strat_parts)
            self._plot_series(
                ax,
                full_ts,
                full_strat,
                label="strategy OOS (chained by year)",
                color="tab:red",
                linewidth=1.8,
            )
        self._style_profit_axis(
            ax,
            "Cross-validation OOS cumulative profit (test years on calendar axis, cumsum chained)",
        )
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        if ts_parts:
            ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
