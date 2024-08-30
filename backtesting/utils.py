# logger.remove()

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


from datetime import timedelta
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_broker import Broker


class BackTestResults:
    def __init__(self, date_start, date_end, wallet=None):
        self.date_start = date_start
        self.date_end = date_end
        self.target_dates = [
            pd.to_datetime(d).date()
            for d in pd.date_range(start=date_start, end=date_end, freq="D")
        ]
        self.daily_hist = pd.DataFrame({"days": self.target_dates})
        self.buy_and_hold = None
        self.tickers = None
        self.wallet = wallet

    def process_backtest(self, backtest_broker: Broker):
        t0 = perf_counter()
        self.wallet = backtest_broker.cfg.wallet if self.wallet is None else self.wallet
        profits = backtest_broker.profits_abs
        dates = [
            pd.to_datetime(pos.close_date).date() for pos in backtest_broker.positions
        ]
        self.tickers = "+".join(set([pos.ticker for pos in backtest_broker.positions]))
        self.process_profits(dates, profits, backtest_broker.fees)
        self.num_years_on_trade = self.compute_n_years(backtest_broker.positions)
        # self.mean_pos_duration = np.array([pos.duration for pos in backtest_broker.positions]).mean()
        return perf_counter() - t0

    def process_profits(self, dates: Iterable, profits: Iterable, fees: Iterable):
        profit_cumsum = np.array(profits).cumsum()
        self.ndeals = len(profits)
        profit_nofees = profit_cumsum + np.array(fees).cumsum()
        self.deal_hist = pd.DataFrame(
            {"dates": dates, "profit": profit_cumsum, "profit_nofees": profit_nofees}
        )
        # self.open_risks = np.array([pos.open_risk for pos in backtest_broker.positions])
        self.update_daily_profit(self._convert_hist(dates, profit_cumsum))
        self.fees = sum(fees)

    def compute_buy_and_hold(self, dates, closes, fuse=False):
        t0 = perf_counter()
        dates = [d.date() for d in pd.to_datetime(dates)]
        yeld_dates = [
            pd.to_datetime(d).date()
            for d in pd.date_range(start=self.date_start, end=self.date_end, freq="M")
        ]
        bh = self._convert_hist(dates, closes, yeld_dates)
        bh = np.hstack([0, ((bh[1:] - bh[:-1]) * self.wallet / bh[:-1])]).cumsum()
        self.daily_hist["buy_and_hold"] = self._convert_hist(yeld_dates, bh)
        self.daily_hist["buy_and_hold_reinvest"] = self._convert_hist(
            dates, closes * self.wallet
        )
        if fuse:
            self.update_daily_profit(
                self.daily_hist["profit"] + self.daily_hist["buy_and_hold"]
            )
        return perf_counter() - t0

    def update_daily_profit(self, daily_profit):
        self.daily_hist["profit"] = daily_profit
        profit_stair, self.metrics = self._calc_metrics(
            self.daily_hist["profit"].values
        )
        self.deposit = self.wallet + self.metrics["loss_max"]
        self.daily_hist["deposit"] = self.deposit - (
            profit_stair - self.daily_hist["profit"].values
        )
        self.metrics["loss_max_rel"] = self.metrics["loss_max"] / self.deposit * 100

    def plot_results(self):
        fig, ax1 = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(10, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = ax1.twinx()
        ax1.plot(
            self.daily_hist.days,
            self.daily_hist.profit,
            linewidth=3,
            color="b",
            alpha=0.6,
        )
        ax1.plot(
            self.deal_hist.dates,
            self.deal_hist.profit,
            linewidth=1,
            color="b",
            alpha=0.6,
        )
        ax1.plot(
            self.deal_hist.dates,
            self.deal_hist.profit_nofees,
            linewidth=1,
            color="r",
            alpha=0.6,
        )
        if "buy_and_hold" in self.daily_hist.columns:
            ax1.plot(
                self.daily_hist.days,
                self.daily_hist.buy_and_hold,
                linewidth=2,
                alpha=0.6,
            )
        ax1.legend(
            ["sum. profit", "profit from strategy", "profit without fees", "buy and hold"]
        )
        ax2.plot(
            self.daily_hist.days,
            self.daily_hist.buy_and_hold_reinvest,
            linewidth=2,
            alpha=0.2,
        )
        ax2.legend([self.tickers])
        # plt.grid("on")

        ax1 = plt.subplot(2, 1, 2)
        ax1.plot(
            self.daily_hist["days"],
            self.daily_hist["deposit"],
            "-",
            linewidth=3,
            alpha=0.3,
        )
        ax1.legend(["deposit"])

        plt.tight_layout()
        plt.savefig("backtest.png")
        plt.show()

    @property
    def final_profit(self):
        return (
            self.daily_hist["profit"].values[-1]
            if len(self.daily_hist["profit"].values) > 0
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

    def _convert_hist(self, dates, vals, target_dates=None):
        if target_dates is None:
            target_dates = self.target_dates
        daily_vals = np.zeros(len(target_dates))
        unbias = False
        darray = np.array(dates)
        for i, date_target in enumerate(target_dates):
            # Select vals records with same day
            for _ in range(10):
                mask = darray == date_target
                if sum(mask):
                    break
                date_target -= timedelta(days=1)
            day_profs = vals[mask]
            # If there are records for currend day, store latest of them, else fill days with no records with latest sored record
            if len(day_profs):
                daily_vals[i] = day_profs[-1]
            elif len(vals):
                daily_vals[i] = daily_vals[i - 1]
            else:
                pass

        if unbias:
            daily_vals = daily_vals - daily_vals[0]
        return daily_vals

    def metrics_from_profit(self, profit_curve):
        loss_max, wait_max = [
            self._calc_metrics(profit_curve)[1][k] for k in ("loss_max", "maxwait")
        ]
        deposit = self.wallet + loss_max
        final_profit_rel = profit_curve[-1] / deposit * 100
        return final_profit_rel / self.num_years_on_trade, wait_max