# logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


from datetime import timedelta
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.backtest_broker import Broker


class BackTestResults:
    def __init__(self, date_start, date_end, wallet=None):
        self.date_start = date_start
        self.date_end = date_end
        # self.target_dates = [
        #     pd.to_datetime(d).date()
        #     for d in pd.date_range(start=date_start, end=date_end, freq="D")
        # ]
        self.daily_hist = None
        self.monthly_hist = None
        self.buy_and_hold = None
        self.tickers = None
        self.wallet = wallet
        self.ndeals = None

    def process_backtest(self, bktest_broker: Broker, leverage=None):
        t0 = perf_counter()
        self.leverage = bktest_broker.cfg["leverage"] if leverage is None else leverage
        self.wallet = bktest_broker.cfg["wallet"] if self.wallet is None else self.wallet

        self.tickers = "+".join(set([pos.ticker for pos in bktest_broker.positions]))
        self.process_profit_hist(bktest_broker.profit_hist)
        self.process_profits(dates=[pd.to_datetime(pos.close_date) for pos in bktest_broker.positions],
                             profits=bktest_broker.profits_abs,
                             fees=bktest_broker.fees_abs)
        self.num_years_on_trade = self.compute_n_years(bktest_broker.positions)
        self.ndeals = len(bktest_broker.positions)
        return perf_counter() - t0

    def process_profit_hist(self, profit_hist: pd.DataFrame):
        profit_hist.set_index("dates", inplace=True)
        self.daily_hist = self.resample_hist(profit_hist, "D", func="last")
        self.daily_hist["profit_csum"] = self.daily_hist["profit_nofees"] - self.daily_hist["fees_csum"]   
        self.process_daily_metrics()
        # self.update_monthly_profit()
        self.fees = profit_hist["fees_csum"][-1]

    def process_profits(self, dates: Iterable, profits: Iterable, fees: Iterable):
        self.ndeals = len(profits)
        if self.ndeals:
            self.deals_hist = pd.DataFrame(
                {"dates": dates, "profits": profits, "fees":fees}
                )       
            self.deals_hist.set_index("dates", inplace=True)
            # self.daily_hist = self.resample_hist(self.deals_hist, "D")
            
            # # add column with cumulative sum of profits
            # self.daily_hist["profit_nofees"] = self.daily_hist["profits"].cumsum()
            # # add column with cumulative sum of fees
            # self.daily_hist["fees_csum"] = self.daily_hist["fees"].cumsum()
            # # add column with cumulative sum of profits without fees
            # self.daily_hist["profit_csum"] = self.daily_hist["profit_nofees"] - self.daily_hist["fees_csum"]   
            self.monthly_hist = self.resample_hist(self.deals_hist, "M")
            # self.process_daily_metrics()
            # self.update_monthly_profit()
            # self.fees = sum(fees)

    def resample_hist(self, hist, period="D", func="sum"):
        target_dates = pd.date_range(start=pd.to_datetime(self.date_start).date(), 
                                     end=pd.to_datetime(self.date_end).date(), 
                                     freq=period) 
               
        hist = hist.resample(period)
        if func == "sum":
            hist = hist.sum() 
            hist = hist.reindex(target_dates, fill_value=0)
        elif func == "last":
            hist = hist.last()
            hist = hist.reindex(target_dates, method="ffill")
        else:
            raise ValueError("func must be 'sum' or 'last'")
        return hist

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
        self.daily_hist["buy_and_hold"] = close_prices * self.wallet/closes[0]
        self.daily_hist["buy_and_hold"].iloc[-1] = self.daily_hist["buy_and_hold"].iloc[-2]
        
        self.daily_hist["unrealized_profit"] = -self.daily_hist["buy_and_hold"].diff() * np.sign(self.daily_hist["profit_nofees"])
        return perf_counter() - t0

    def process_daily_metrics(self):
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
        temp_df.index = pd.to_datetime(temp_df.index)

        # Calculate monthly profits
        self.monthly_hist = temp_df['profit'].resample('M').sum()
        self.monthly_hist = self.monthly_hist.reset_index()
        self.monthly_hist.columns = ["days", "profit"]
        
    def plot_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # First subplot
        ax1.plot(
            self.daily_hist.index,
            self.daily_hist.profit_csum,
            linewidth=3,
            color="b",
            alpha=0.6,
        )

        ax1.plot(
            self.daily_hist.index,
            self.daily_hist.profit_nofees,
            linewidth=1,
            color="r",
            alpha=0.6,
        )
        
        legend_ax1 = ["profit", "profit without fees"]
        
        if "buy_and_spend" in self.monthly_hist.columns:
            ax1.plot(
                self.monthly_hist.index,
                self.monthly_hist["buy_and_spend"],
                linewidth=2,
                alpha=0.6,
            )
            legend_ax1.append("buy & spend")

        # Create a secondary y-axis for buy_and_hold_reinvest
        if "buy_and_hold" in self.daily_hist.columns:
            # ax1 = ax1.twinx()
            ax1.plot(
                self.daily_hist.index,
                self.daily_hist["buy_and_hold"],
                linewidth=2,
                color="g",
                alpha=0.6,
            )
            legend_ax1.append("buy & hold")

        # Second subplot
        ax2.plot(
            self.daily_hist.index,
            self.daily_hist["deposit"],
            "-",
            linewidth=3,
            alpha=0.3,
        )
        

        # Third subplot
        ax3.bar(
            self.monthly_hist.index,
            self.monthly_hist["profits"],
            width=20,
            color="g",
            alpha=0.6,
        )

        ax1.legend(legend_ax1)    
        ax2.legend(["deposit"])    
        ax3.legend(["monthly profit"])

        plt.tight_layout()
        plt.savefig("backtest.png")
        plt.show()

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
        return int(self.ndeals / max(1, self.num_years_on_trade) / 12)

    def compute_n_years(self, positions):
        if len(positions) == 0:
            return 0
        d0 = max(np.datetime64(self.date_start), positions[0].open_date)
        d1 = min(np.datetime64(self.date_end), positions[-1].close_date)
        return (d1 - d0).astype("timedelta64[M]").item() / 12

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