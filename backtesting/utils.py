# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


from functools import cached_property
from time import perf_counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from backtesting.backtest_broker import Broker
from common.type import to_datetime


class BackTestResults:
    def __init__(self):
        self.daily_hist = pd.DataFrame()
        self.monthly_hist = pd.DataFrame()
        self.buy_and_hold = None
        self.tickers = set()
        self.wallet = 0
        self.deposit = 0
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

    def add(self, bktest_broker: Broker):
        self.ndeals += len(bktest_broker.positions)
        self.wallet += bktest_broker.wallet
        self.tickers.update([pos.ticker for pos in bktest_broker.positions])
        self.positions.extend(bktest_broker.positions)
        
        daily_hist, monthly_hist = self.process_profit_hist(bktest_broker.profit_hist)
        
        if self.daily_hist.empty:
            self.daily_hist = daily_hist
        else:
            # Align the indices of both DataFrames
            all_dates = self.daily_hist.index.union(daily_hist.index)
            
            # Reindex both DataFrames to include all dates
            self_reindexed = self.daily_hist.reindex(all_dates)
            daily_hist_reindexed = daily_hist.reindex(all_dates)
            
            # Fill NaNs with the last value for each DataFrame
            self_reindexed.fillna(method='ffill', inplace=True)
            daily_hist_reindexed.fillna(method='ffill', inplace=True)
            
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
        assert "dates" in profit_hist.columns, "profit_hist must have 'dates' column"
        profit_hist.set_index("dates", inplace=True)
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
        
        return (hist.resample(period)
                   .agg(agg_method)
                   .reindex(target_dates, method=fill_method if func == "last" else None,
                          fill_value=fill_method if func == "sum" else None))



    def compute_buy_and_hold(self, dates: np.ndarray, closes: np.ndarray):
        """
        Compute buy and hold strategy for the given dates and closes prices. Use self.resample_hist method to resample data to monthly frequency.
        
        Output pandas DataFrame with index "dates" and columns: "profits", "buy_and_hold", "buy_and_hold_reinvest".
        dates - dates of first day of every month.
        profits - profits from the strategy on every month.
        buy_and_hold - Buy at the first day and sell on the last. 
        """
        t0 = perf_counter()
        # create datafreame from closes and dates as index
        df = pd.DataFrame({"price": closes}, index=dates)
        close_prices = df.resample("D").last()["price"]
        self.daily_hist["buy_and_hold"] = (close_prices - close_prices[0]) * self.wallet/close_prices[0]
        if self.daily_hist.shape[0] > 1:
            self.daily_hist["buy_and_hold"].iloc[-1] = self.daily_hist["buy_and_hold"].iloc[-2]

        self.daily_hist["unrealized_profit"] = -self.daily_hist["buy_and_hold"].diff() * np.sign(self.daily_hist["profit_csum_nofees"])
        return perf_counter() - t0

    def eval_daily_metrics(self):
        profit_stair, self.metrics = self._calc_metrics(
            self.daily_hist["profit_csum"].values
        )
        self.deposit = self.wallet + self.metrics["loss_max"]
        self.daily_hist["deposit"] = self.deposit - (
            profit_stair - self.daily_hist["profit_csum"].values
        )
        self.metrics["loss_max_rel"] = self.metrics["loss_max"] / self.deposit * 100

    def update_monthly_profit(self):
        # Create a temporary DataFrame with a DatetimeIndex for resampling
        temp_df = self.daily_hist.copy()
        temp_df.set_index("days", inplace=True)
        temp_df.index = to_datetime(temp_df.index)

        # Calculate monthly profits
        self.monthly_hist = temp_df['profit'].resample('M').sum()
        self.monthly_hist = self.monthly_hist.reset_index()
        self.monthly_hist.columns = ["days", "profit"]

    def relative2wallet(self, data: pd.Series):
        return data/self.wallet*100

    def add_profit_curve(self, days: pd.Index, values: pd.Series, name: str, color: str, linewidth: float, alpha: float):
        self.fig.axes[0].plot(days, values, linewidth=linewidth, color=color, alpha=alpha)
        self.legend_ax1.append(name)
        
    def add_from_other_results(self, btest_results: "BackTestResults", color: str, linewidth: float = 1, alpha: float = 0.5):
        self.add_profit_curve(btest_results.daily_hist.index, btest_results.relative2wallet(btest_results.daily_hist["profit_csum"]), 
                              btest_results.tickers_set, color, linewidth, alpha)

    def plot_results(self, title: Optional[str] = None, plot_profit_without_fees: bool = True):
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        if title:
            self.fig.suptitle(title, fontsize=16)
            self.fig.subplots_adjust(top=0.9)
        
        self.add_profit_curve(self.daily_hist.index, self.relative2wallet(self.daily_hist["profit_csum"]), 
                              self.tickers_set, color="b", linewidth=3, alpha=0.5)
        if plot_profit_without_fees:
            self.add_profit_curve(self.daily_hist.index, self.relative2wallet(self.daily_hist["profit_csum_nofees"]), 
                                  f"{self.tickers_set } without fees", color="b", linewidth=1, alpha=0.5)

        if "buy_and_hold" in self.daily_hist.columns:
            ax1.plot(
                self.daily_hist.index,
                self.daily_hist["buy_and_hold"]/self.wallet*100,
                linewidth=2,
                color="g",
                alpha=0.6,
            )
            self.legend_ax1.append("buy & hold")
            
        ax1.set_ylabel("fin result, %")

        ax2.plot(
            self.daily_hist.index,
            self.daily_hist["deposit"]/self.wallet*100,
            "-",
            linewidth=3,
            alpha=0.3,
        )
        ax2.set_ylabel("deposit, %")

        ax3.bar(
            self.monthly_hist.index,
            self.monthly_hist["finres"]/self.wallet*100,
            width=20,
            color="g",
            alpha=0.6,
        )
        ax3.set_ylabel("monthly profit, %")

        plt.tight_layout()
        
    def save_fig(self, save_path: Optional[str] = "_last_backtest.png"):
        self.fig.axes[0].legend(self.legend_ax1)
        if self.fig is not None:
            self.fig.savefig(save_path)

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

    @staticmethod
    def _calc_metrics(ts):
        ymax = ts[0]
        twait = 0
        twaits = []
        h = np.zeros(len(ts))
        for i, y in enumerate(ts):
            if y >= ymax:
                ymax = y
                if twait > 0:
                    # print(t, twait ,ymax)
                    twaits.append(twait)
                    twait = 0
            else:
                twait += 1
            h[i] = ymax
        max_loss = (h - ts).max()
        twaits.append(twait)
        twaits = np.array(twaits) if len(twaits) else np.array([len(ts)])
        twaits.sort()
        # lin_err = sum(np.abs(ts - np.arange(0, ts[-1], ts[-1]/len(ts))[:len(ts)]))
        # lin_err /= len(ts)*ts[-1]
        metrics = {
            "maxwait": twaits.max(),  # [-5:].mean(),
            "recovery": ts[-1] / max_loss if max_loss else -1,
            "loss_max": max_loss,
        }
        return h, metrics

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
    
    def normalized_profits_entropy(self, positions_finresults):
        """
        Вычисляет нормированную энтропию распределения прибылей.
        
        Аргументы:
            profits (list или np.array): массив значений прибылей по сделкам.
            
        Возвращает:
            Нормированная энтропия в диапазоне [0, 1], 
            где 1 соответствует равномерному распределению, 0 — концентрации прибыли в одном элементе.
        """
        profits = positions_finresults[positions_finresults > 0].values
        
        # Если суммарная прибыль равна 0, энтропия неопределена
        total_profit = np.sum(profits)
        if total_profit == 0:
            raise ValueError("Суммарная прибыль равна 0, энтропия не может быть вычислена.")
        
        # Вычисляем доли (веса) каждой сделки
        weights = profits / total_profit
        
        # Для вычисления логарифма учитываем только ненулевые веса, чтобы избежать ошибки log(0)
        nonzero_weights = weights[weights > 0]
        
        # Вычисляем энтропию: H = -sum(w_i * ln(w_i))
        H = -np.sum(nonzero_weights * np.log(nonzero_weights))
        
        # Нормируем энтропию относительно максимума ln(N), где N - число сделок
        N = len(profits)
        H_max = np.log(N)
        normalized_H = H / H_max if H_max > 0 else 1.0
        
        return normalized_H

    @property
    def fees(self) -> float:
        """Total accumulated fees from trading"""
        if self.daily_hist is None:
            return 0.0
        return self.daily_hist["fees_csum"].iloc[-1]
        
    def print_results(self, cfg: Optional[dict] = None, expert_name: Optional[str] = None) -> None:
        """Print formatted backtest results to the log."""
        
        def sformat(nd): return "{:>30}: {:>5.@f}".replace("@", str(nd))

        print()
        if cfg is not None and expert_name is not None:
            logger.info(f"{cfg['symbol'].ticker}-{cfg['period']}-{cfg['hist_size']}: {expert_name}")

        logger.info("-" * 40)
        logger.info(sformat(0).format("APR", self.APR) + f" %")
        logger.info(
            sformat(0).format("FINAL PROFIT", self.final_profit_rel)
            + f" %"
            + f" ({self.fees/(self.final_profit*100 + 1e-6):.1f}% fees)"
        )
        logger.info(
            sformat(1).format("DEALS/MONTH", self.ndeals_per_month)
            + f"   ({self.ndeals} total)"
        )
        logger.info(sformat(0).format(
            "MAXLOSS", self.metrics["loss_max_rel"]) + " %")
        logger.info(sformat(0).format(
            "RECOVRY FACTOR", self.metrics["recovery"]))
        logger.info(sformat(0).format(
            "MAXWAIT", self.metrics["maxwait"]) + " days")

