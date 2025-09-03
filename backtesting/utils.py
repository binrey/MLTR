# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from io import BytesIO

from backtesting.backtest_broker import Broker, TradeHistory
from common.type import VolEstimRule, to_datetime
from trade.utils import Position


class Metrics:
    def __init__(self, dates, profit_curve, realized_pnl_withfees):
        # Calculate running maximum (all-time high) of profit_curve
        self.ath_curve = np.maximum.accumulate(realized_pnl_withfees)
        
        max_period, price_at_max_period = 0, 0
        max_period_start = dates[0]
        max_period_end = dates[0]
        
        curr_period = 0
        curr_start = dates[0]
        
        for i in range(1, len(self.ath_curve)):
            if self.ath_curve[i] <= self.ath_curve[i-1]:
                curr_period += 1
            else:
                if curr_period > max_period:
                    max_period = curr_period
                    price_at_max_period = self.ath_curve[i-1]
                    max_period_start = curr_start
                    max_period_end = dates[i-1]
                curr_period = 0
                curr_start = dates[i]

        # Check final period
        if curr_period > max_period:
            max_period = curr_period
            price_at_max_period = self.ath_curve[-1]
            max_period_start = curr_start
            max_period_end = dates[-1]
        
        self.max_period = max_period
        self.price_at_max_period = price_at_max_period
        self.max_period_start = max_period_start
        self.max_period_end = max_period_end
        
        self.drawdown_curve = self.ath_curve - np.minimum(profit_curve, realized_pnl_withfees)
        self.max_drawdown = np.max(self.drawdown_curve)
        self.recovery_factor = profit_curve[-1] / self.max_drawdown if self.max_drawdown > 0 else -1


class BackTestResults:
    def __init__(self):
        self.daily_hist = pd.DataFrame()
        self.monthly_hist = pd.DataFrame()
        self.leverage = 1
        self.tickers = set()
        self.deposit = None
        self.vol_estim_rule: Optional[VolEstimRule] = None
        self.ndeals = 0
        self.positions = []
        self.fig = None
        self.legend_ax1 = []
        self.metrics = {}
        
    @property
    def date_start(self):
        return self.daily_hist.index.min()
    
    @property
    def date_end(self):
        return self.daily_hist.index.max()
    
    @property
    def num_years_on_trade(self):
        return (self.date_end - self.date_start).days / 365
    
    @property
    def tickers_set(self) -> str:
        return "+".join(self.tickers)

    def process(self):
        self.eval_daily_metrics()

    def add(self, profit_hist: TradeHistory, same_deposit: bool = True):
        positions = profit_hist.positions
        self.ndeals += len(positions)
        self.tickers.update([pos.ticker for pos in positions])
        self.positions.extend(positions)

        deposit = profit_hist.deposit
        if self.deposit is None:
            self.deposit = deposit
        else:
            if not same_deposit:
                self.deposit += deposit
            elif profit_hist.volume_control.rule == VolEstimRule.FIXED_POS_COST:
                self.deposit += deposit - profit_hist.wallet

        self.leverage = profit_hist.leverage

        daily_hist, monthly_hist = self.process_profit_hist(profit_hist.df)
        
        if self.daily_hist.empty:
            self.daily_hist = daily_hist
        else:
            # Align the indices of both DataFrames
            all_dates = self.daily_hist.index.union(daily_hist.index)
            
            # Reindex both DataFrames to include all dates
            self_reindexed = self.daily_hist.reindex(all_dates)
            daily_hist_reindexed = daily_hist.reindex(all_dates)
            
            # Fill NaNs with the last value for each DataFrame
            self_reindexed.ffill(inplace=True)
            daily_hist_reindexed.ffill(inplace=True)
            
            # Add the DataFrames
            self.daily_hist = self_reindexed.add(daily_hist_reindexed, fill_value=0)
            
        if self.monthly_hist.empty:
            self.monthly_hist = monthly_hist
        else:
            # Apply similar logic for monthly_hist
            all_months = self.monthly_hist.index.union(monthly_hist.index)
            
            # Reindex both DataFrames to include all months
            self_monthly_reindexed = self.monthly_hist.reindex(all_months)
            monthly_hist_reindexed = monthly_hist.reindex(all_months)
            
            # Fill NaNs with zeros for monthly data
            self_monthly_reindexed.fillna(0, inplace=True)
            monthly_hist_reindexed.fillna(0, inplace=True)
            
            # Add the DataFrames
            self.monthly_hist = self_monthly_reindexed.add(monthly_hist_reindexed, fill_value=0)            

    def process_profit_hist(self, profit_hist: pd.DataFrame):
        if profit_hist.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame()
        if "dates" in profit_hist.columns:
            profit_hist.set_index("dates", inplace=True)
        elif profit_hist.index.name != "dates":
            "profit_hist must have 'dates' column or dates in index"
        daily_hist = self.resample_hist(profit_hist, "D", func="last")
        daily_hist["profit_csum"] = daily_hist["profit_csum_nofees"] - daily_hist["fees_csum"]
        daily_hist["finres"] = daily_hist["profit_csum"].diff().fillna(0)
        monthly_hist = self.resample_hist(daily_hist, "M")
        return daily_hist, monthly_hist

    def resample_hist(self, hist, period="D", func="sum"):
        target_dates = pd.date_range(start=to_datetime(hist.index.min()).date(),
                                     end=to_datetime(hist.index.max()).date(),
                                     freq=period)
        
        agg_method = "sum" if func == "sum" else "last"
        fill_method = {"sum": 0, "last": "ffill"}[func]
        
        hist_resampled = (hist.resample(period)
                   .agg(agg_method)
                   .reindex(target_dates, method=fill_method if func == "last" else None,
                          fill_value=fill_method if func == "sum" else None))
        return hist_resampled

    def eval_daily_metrics(self):
        self.metrics = Metrics(
            self.daily_hist.index,
            self.daily_hist["profit_csum"].values,
            self.daily_hist["realized_pnl_withfees"].values,
        )
        self.daily_hist["deposit"] = self.deposit - self.metrics.drawdown_curve

    def update_monthly_profit(self):
        # Create a temporary DataFrame with a DatetimeIndex for resampling
        temp_df = self.daily_hist.copy()
        temp_df.set_index("days", inplace=True)
        temp_df.index = to_datetime(temp_df.index)

        # Calculate monthly profits
        self.monthly_hist = temp_df['profit'].resample('M').sum()
        self.monthly_hist = self.monthly_hist.reset_index()
        self.monthly_hist.columns = ["days", "profit"]

    def relative2deposit(self, data: pd.Series):
        return data/self.deposit*100

    def add_profit_curve(self, days: pd.Index, values: pd.Series, name: str, color: str, linewidth: float, alpha: float):
        self.fig.axes[0].plot(days, values, linewidth=linewidth, color=color, alpha=alpha)
        self.legend_ax1.append(name)

    def add_from_other_results(self, btest_results: "BackTestResults", color: str, linewidth: float = 1, alpha: float = 0.5, use_relative: bool = True):
        data = btest_results.relative2deposit(btest_results.daily_hist["profit_csum"]) if use_relative else btest_results.daily_hist["profit_csum"]
        self.add_profit_curve(btest_results.daily_hist.index, data, 
                              btest_results.tickers_set, color, linewidth, alpha)

    def plot_validation(self, title: Optional[str] = None, y_label="Fin result, %", use_relative: bool = True):
        self.fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        if title:
            self.fig.suptitle(title, fontsize=16)
            self.fig.subplots_adjust(top=0.9)

        if use_relative and y_label == "Fin result, %":
            ax1.set_ylabel("Fin result, %")
        else:
            ax1.set_ylabel(y_label)


    def plot_results(self, title: Optional[str] = None, plot_profit_without_fees: bool = True, use_relative: bool = True):
        self.fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        if title:
            self.fig.suptitle(title, fontsize=16)
            self.fig.subplots_adjust(top=0.9)
        
        if use_relative:
            ax1.set_ylabel("fin result, %")
            ax2.set_ylabel("position cost, %")
            ax3.set_ylabel("deposit, %")
            ax4.set_ylabel("monthly profit, %")
        else:
            ax1.set_ylabel("fin result")
            ax2.set_ylabel("position cost")
            ax3.set_ylabel("deposit")
            ax4.set_ylabel("monthly profit")

        # Add vertical grid lines to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        # Hide x-axis tick labels for all axes except ax1
        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, top=False)

        # -------------------------------------------

        # Helper function to apply relative conversion if needed
        def apply_relative(data):
            return self.relative2deposit(data) if use_relative else data

        # self.add_profit_curve(self.daily_hist.index,
        #                       apply_relative(self.daily_hist["profit_csum"]),
        #                       self.tickers_set,
        #                       color="b",
        #                       linewidth=3,
        #                       alpha=0.5)
        self.add_profit_curve(self.daily_hist.index,
                              apply_relative(self.daily_hist["realized_pnl_withfees"]),
                              self.tickers_set,
                              color="b",
                              linewidth=3,
                              alpha=0.5)
        if plot_profit_without_fees:
            self.add_profit_curve(self.daily_hist.index,
                                  apply_relative(self.daily_hist["profit_csum_nofees"]),
                                  f"{self.tickers_set } without fees",
                                  color="b",
                                  linewidth=1,
                                  alpha=0.5)

        # Plot max ATH period
        ax1.plot([self.metrics.max_period_start, self.metrics.max_period_end],
                 [apply_relative(self.metrics.price_at_max_period)]*2,
                 color="r", linewidth=2, alpha=0.5, linestyle="--")
        ax1.text(self.metrics.max_period_start,
                 apply_relative(self.metrics.price_at_max_period + self.fig.axes[0].get_ylim()[1] * 0.01),
                 f"{self.metrics.max_period:.0f} days",
                 color="r",
                 fontsize=12)

        ax2.plot(
            self.daily_hist.index,
            apply_relative(self.daily_hist["pos_cost"]),
            "-",
            linewidth=3,
            alpha=0.3,
        )
        if use_relative:
            ax2.set_ylim(0, 100*self.leverage)

        assert "deposit" in self.daily_hist.columns, "deposit column must be in daily_hist, do eval_daily_metrics before plotting"
        ax3.plot(
            self.daily_hist.index,
            apply_relative(self.daily_hist["deposit"]),
            "-",
            linewidth=3,
            alpha=0.3,
        )

        ax4.bar(
            self.monthly_hist.index,
            apply_relative(self.monthly_hist["finres"]),
            width=20,
            color="g",
            alpha=0.6,
        )

    def save_fig(self, save_path: Optional[str] = "_last_backtest.png"):
        self.fig.axes[0].legend(self.legend_ax1)
        plt.tight_layout()
        if self.fig is not None:
            self.fig.savefig(save_path)

    def show_fig(self):
        if self.fig is None:
            return
        # Ensure legend and layout are finalized before rendering
        self.fig.axes[0].legend(self.legend_ax1)
        plt.tight_layout()
        # Render to buffer and display with PIL to avoid backend show issues
        buffer = BytesIO()
        self.fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        Image.open(buffer).show()
        buffer.close()
        plt.close(self.fig)

    @property
    def final_profit(self):
        if self.daily_hist is not None:
            return (
                self.daily_hist["profit_csum"].values[-1]
                if len(self.daily_hist["profit_csum"].values) > 0
                else 0
            )

    @property
    def final_profit_rel(self):
        return self.final_profit / self.deposit * 100

    @property
    def APR(self):
        return self.final_profit_rel / self.num_years_on_trade if self.num_years_on_trade > 0 else 0

    @property
    def ndeals_per_year(self):
        return int(self.ndeals / self.num_years_on_trade)

    @property
    def ndeals_per_month(self):
        return self.ndeals / max(1, self.num_years_on_trade) / 12

    def metrics_from_profit(self, profit_curve):
        loss_max, wait_max = [
            self._calc_metrics(profit_curve)[1][k] for k in ("loss_max", "maxwait")
        ]
        deposit = self.wallet + loss_max
        final_profit_rel = profit_curve[-1] / deposit * 100
        return final_profit_rel / self.num_years_on_trade, wait_max

    def profits_histogram(self, positions_finresults):
        npos_by_bins, bin_edges = np.histogram(positions_finresults)
        finres_by_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        return npos_by_bins, finres_by_bins

    @property
    def fees(self) -> float:
        """Total accumulated fees from trading"""
        if self.daily_hist is None:
            return 0.0
        return self.daily_hist["fees_csum"].iloc[-1]

    def print_results(self, cfg: Optional[dict] = None, expert_name: Optional[str] = None, use_relative: bool = True) -> None:
        """Print formatted backtest results to the log."""

        def sformat(nd): return "{:>30}: {:>5.@f}".replace("@", str(nd))

        print()
        if cfg is not None and expert_name is not None:
            print(f"{cfg['symbol'].ticker}-{cfg['period']}-{cfg['hist_size']}: {expert_name}")

        print("-" * 40)

        # Helper function to apply relative conversion if needed
        def apply_relative(value):
            return self.relative2deposit(value) if use_relative else value

        def get_unit_suffix():
            return " %" if use_relative else ""

        print(sformat(0).format("APR", self.APR) + get_unit_suffix())
        print(
            sformat(0).format("FINAL PROFIT", self.final_profit_rel if use_relative else self.final_profit)
            + get_unit_suffix()
            + f" ({self.fees/(self.final_profit + 1e-6)*100:.1f}% fees)"
        )
        print(
            sformat(1).format("DEALS/MONTH", self.ndeals_per_month)
            + f"   ({self.ndeals} total)"
        )
        print(sformat(0).format(
            "MAXLOSS", apply_relative(self.metrics.max_drawdown)) + get_unit_suffix())
        print(sformat(1).format(
            "RECOVRY FACTOR", self.metrics.recovery_factor))
        print(sformat(0).format(
            "MAXWAIT", self.metrics.max_period) + " days")
