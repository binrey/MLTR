import itertools
import multiprocessing
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

from backtesting.backtest_broker import TradeHistory
from backtesting.utils import BackTestResults, Metrics
from common.type import Symbol
from common.utils import date2str
from data_processing import PULLERS
from trade.backtest import BackTest
from trade.backtest import launch as backtest_launch
from trade.utils import Position

# logger.remove()
# logger.add(sys.stderr, level="INFO")
# logger = logger.bind(module="data_processing.dataloading")
# logger.disable("data_processing.dataloading")

pd.set_option('display.max_colwidth', 1028)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def plot_daily_balances_with_av(btests: List[BackTestResults], test_ids: List[int], profit_av: np.ndarray, metrics_av: List[Tuple[str, float]]):
    """
    Plots daily balances for multiple backtest results and their average.
    """
    legend = []
    for test_id in test_ids:
        btest = btests[test_id]
        plt.plot(btest.daily_hist.index, btest.daily_hist["profit_csum"])
        legend.append(
            f"{btest.tickers} profit={btest.final_profit:.0f} ({btest.ndeals}) APR={btest.APR:.2f} mwait={btest.metrics['maxwait']:.0f}")
    plt.plot(btests[0].daily_hist.index, profit_av, linewidth=3, color="black")
    legend_item = f"AV profit={profit_av[-1]:.0f},"
    for (name, val) in metrics_av:
        legend_item += f" {name}={val:.2f}"
    legend.append(legend_item)
    plt.legend(legend)
    plt.grid("on")
    plt.tight_layout()


def collect_config_combinations(config):
    """
    Recursively enumerates all possible parameter combinations in `config`,
    where each key can hold either:
        - A single value
        - A list of values
        - A nested dictionary following the same pattern

    Returns:
        A list of fully expanded configuration dictionaries (the Cartesian 
        product of all possible paths through nested lists/dictionaries).
    """
    # Base case: if config is not a dict, it is a single leaf-value
    if not isinstance(config, dict):
        return [config]

    # For each key in the dict, get all expanded possibilities (list of expansions)
    expanded_subconfigs = {}  # key -> list of expanded variants
    for key, value in config.items():
        # If value is a list, expand each element of that list
        if isinstance(value, list):
            all_expanded_for_key = []
            for item in value:
                all_expanded_for_key.extend(collect_config_combinations(item))
            expanded_subconfigs[key] = all_expanded_for_key
        # If value is a dictionary or a single value, expand it directly
        else:
            expanded_subconfigs[key] = collect_config_combinations(value)

    # Now do a Cartesian product across all keys to get final expansions
    all_keys = list(expanded_subconfigs.keys())
    list_of_expanded_lists = [expanded_subconfigs[k] for k in all_keys]

    all_combinations = []
    for combo in itertools.product(*list_of_expanded_lists):
        # combo is a tuple with one expanded item per key
        # Merge them into a single config dictionary
        cfg_variant = {}
        for i, item in enumerate(combo):
            cfg_variant[all_keys[i]] = item
        all_combinations.append(cfg_variant)

    return all_combinations


class Optimizer:
    """
    Optimizer class for optimizing trading strategies.
    """
    @dataclass
    class OptimizationResults:
        """
        Results of the optimization process.
        """
        opt_summary: pd.DataFrame  # Optimization summary DataFrame
        configs_by_run: Dict  # Configurations
        positions_by_run: List[Position]
        score_name: str = "APR"

        def __post_init__(self):
            self.sort_by(self.score_name)

        def __str__(self) -> str:
            return tabulate(self.opt_summary, headers='keys', tablefmt='psql', showindex=True, floatfmt='.2f')

        def __repr__(self) -> str:
            return self.__str__()

        def sort_by(self, score_name: str = "APR"):
            self.opt_summary = self.opt_summary.sort_values(
                by=[score_name], ascending=False)

        def apply_filters(self, max_ndeals_per_month: int = 4, min_ndeals_per_month: int = 1):
            self.opt_summary = self.opt_summary[
                (self.opt_summary["ndeals_per_month"] < max_ndeals_per_month) &
                (self.opt_summary["ndeals_per_month"] > min_ndeals_per_month)
            ]

        @property
        def top_configuration_id(self):
            return " ".join(map(str, [
                self.opt_summary.loc[self.top_run_id]["symbol"],
                f"{'hist_size'}:{self.opt_summary.loc[self.top_run_id]['hist_size']}",
                f"{'decision_maker'}:{self.opt_summary.loc[self.top_run_id]['decision_maker']}",
                f"{'trailing_stop'}:{self.opt_summary.loc[self.top_run_id]['trailing_stop']}"
            ]))

        @property
        def top_run_id(self) -> int:
            return self.opt_summary.index[0]

        @property
        def best_config(self) -> Dict[str, Any]:
            return self.configs_by_run[self.top_run_id]

        @property
        def best_score(self) -> float:
            return self.opt_summary.iloc[0][self.score_name]

    def __init__(self, optim_cfg: Dict[str, Any], results_dir="results/optimization", sortby: str = "APR"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = Path(".cache/backtests")
        self.sortby = sortby
        self.cfg = optim_cfg
        self.cfgs = []
        self.bt_results: List[TradeHistory] = []

    def _get_cache_subdir(self, cfg):
        """Get the appropriate cache subdirectory for a configuration."""
        if isinstance(cfg['decision_maker'], list):
            if len(cfg['decision_maker']) == 1:
                decision_maker_type = cfg['decision_maker'][0]['type'].__name__
            else:
                raise ValueError(
                    "Multiple decision makers are not supported yet")
        else:
            decision_maker_type = cfg['decision_maker']['type'].__name__
        if isinstance(cfg['symbol'], list):
            if len(cfg['symbol']) == 1:
                symbol_ticker = cfg['symbol'][0].ticker
            else:
                symbol_ticker = ""
        else:
            symbol_ticker = cfg['symbol'].ticker
        return self.cache_dir / decision_maker_type / symbol_ticker

    def clear_cache(self):
        if self.cache_dir.exists():
            rmtree(self.cache_dir)

    def backtest_process(self, args):
        num, cfg = args
        logger.debug(f"start backtest {num}: {cfg}")
        locnum = 0
        while True:
            PULLERS["bybit"](**cfg)
            backtest_trading = BackTest(cfg)
            backtest_trading.initialize()
            backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
            # bt_res = backtest_trading.postprocess()
            locnum += 1

            # Create cache subdirectory
            cache_subdir = self._get_cache_subdir(cfg)
            cache_subdir.mkdir(exist_ok=True, parents=True)

            # Save backtest results
            cache_file = cache_subdir / \
                f"btest.{num + locnum/100:05.2f}.pickle"
            with open(cache_file, "wb") as f:
                pickle.dump((cfg, backtest_trading.session.profit_hist), f)
            break

    def pool_handler(self, optim_cfg):
        # Only remove results directory, not cache
        if self.results_dir.exists():
            rmtree(self.results_dir)

        # Create cache dir if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        ncpu = multiprocessing.cpu_count()
        logger.info(f"Number of cpu : {ncpu}")

        cfgs = collect_config_combinations(optim_cfg)
        logger.info(f"optimization steps number: {len(cfgs)}")

        cfgs = [(i, cfg) for i, cfg in enumerate(cfgs)]
        p = Pool(ncpu)
        return p.map(self.backtest_process, cfgs)

    def optimize(self, optim_cfg) -> OptimizationResults:
        symbol = optim_cfg["symbol"]
        # results_dir = self.results_dir / f"{symbol.value}"

        # Load and validate results
        self.cfgs, self.bt_results = self._load_backtest_results(optim_cfg)
        if not self.cfgs:
            raise ValueError(f"No backtest results found for {symbol.ticker}")

        # Validate dates
        self._validate_backtest_dates()
        # Find best multistrategy
        multistrategy_test = self.find_best_multistrategy(num_backtests=4)
        # Generate optimization summary
        opt_summary = self._generate_optimization_summary()

        # Save and plot results
        # self._plot_optimization_results(opt_summary, symbol, period)

        return self.OptimizationResults(score_name=self.sortby,
                                        opt_summary=opt_summary,
                                        configs_by_run=self.cfgs,
                                        positions_by_run=self.bt_results)

    def run_backtests(self) -> None:
        """
        Run backtests for the given optimization configuration.

        Args:
            optim_cfg: Configuration dictionary for optimization
        """
        t0 = time()
        self.pool_handler(self.cfg)
        logger.info(f"Optimization time: {time() - t0:.1f} sec\n")

    def _load_backtest_results(self, cfg):
        """Load backtest results from pickle files."""
        cfgs, btests = [], []
        cache_dir = self._get_cache_subdir(cfg)
        # Get all subdirectories in cache
        if not cache_dir.exists():
            return cfgs, btests

        # Get all pickle files
        pickle_files = sorted(cache_dir.rglob("*.pickle"))
        
        # Recursively find all pickle files with progress bar
        for p in tqdm(pickle_files, desc="Loading backtest results"):
            try:
                with open(p, "rb") as f:
                    cfg, trade_history = pickle.load(f)
                cfgs.append(cfg)
                btests.append(trade_history)
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                logger.warning(f"Failed to load cache file {p}: {e}")
                continue

        return cfgs, btests

    def _validate_backtest_dates(self) -> None:
        """Validate that all backtests have the same start date."""
        start_dates = set([btest.date_start for btest in self.bt_results])
        assert len(start_dates) == 1, f"Inconsistent backtest start dates. \
            Please adjust configuration to start from {date2str(max(start_dates))}"


    def _generate_optimization_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of optimization results."""
        opt_summary = defaultdict(list)

        # Extract configuration parameters
        cfg_keys = list(self.cfgs[0].keys())
        cfg_keys.remove("no_trading_days")

        for k in cfg_keys:
            for cfg in self.cfgs:
                v = cfg[k]
                if isinstance(v, dict):
                    description = []
                    for kk, vv in v.items():
                        if kk == "type":
                            description.append(f"{vv.__name__}")
                        elif isinstance(vv, Enum):
                            description.append(f"{kk}:{vv.value}")
                        else:
                            description.append(f"{kk}:{vv}")
                    opt_summary[k].append("|".join(description))
                else:
                    if isinstance(v, Symbol):
                        v = v.ticker
                    opt_summary[k].append(v)

        # Add metrics
        for trade_history in self.bt_results:
            btest = BackTestResults()
            btest.add(trade_history)
            btest.eval_daily_metrics()
            opt_summary["APR"].append(btest.APR)
            opt_summary["final_profit"].append(btest.final_profit)
            opt_summary["ndeals_per_month"].append(btest.ndeals_per_month)
            opt_summary["loss_max_rel"].append(btest.relative2deposit(btest.metrics.max_drawdown))
            opt_summary["recovery"].append(btest.metrics.recovery_factor)
            opt_summary["maxwait"].append(btest.metrics.max_period)

        # Convert to DataFrame and clean up
        opt_summary = pd.DataFrame(opt_summary)

        # Remove constant columns except symbol
        if opt_summary.shape[0] > 1:
            for k in list(opt_summary.columns):
                if k != "symbol" and len(set(map(str, opt_summary[k]))) == 1:
                    del opt_summary[k]

        opt_summary.sort_values(by=["APR"], ascending=False, inplace=True)
        return opt_summary

    def find_best_multistrategy(self, num_backtests: int = 10) -> BackTestResults:
        """
        Find the combination of backtests provided best APR.
        """
        metrics = {}
        for i, trade_history in enumerate(self.bt_results):
            btest = BackTestResults()
            btest.add(trade_history)
            btest.eval_daily_metrics()
            metrics[i] = btest.metrics.recovery_factor
        sorted_btest_ids = sorted(metrics, key=lambda x: metrics[x], reverse=True)
        btest_selected = sorted_btest_ids[0]
        best_setup, best_metric = [btest_selected], metrics[btest_selected]
        print(f"+ {btest_selected}: {self.bt_results[btest_selected].ticker} {best_metric:.2f}")
        
        for adding_iter in range(1, num_backtests):
            btest_selected = None
            print(f"{adding_iter} {best_setup} {best_metric:.2f}")
            for btest2add in sorted_btest_ids:
                if btest2add in best_setup:
                    continue
                multistrategy_test = BackTestResults()
                for btest_id in best_setup + [btest2add]:
                    trade_hist = self.bt_results[btest_id]
                    multistrategy_test.add(trade_hist, same_deposit=False)
                multistrategy_test.eval_daily_metrics()
                
                if multistrategy_test.metrics.recovery_factor > best_metric:
                    best_metric = multistrategy_test.metrics.recovery_factor
                    btest_selected = btest2add
                    print(f"+ {btest_selected}: {trade_hist.ticker} {best_metric:.2f}")
            if btest_selected is None:
                break
            else:
                best_setup.append(btest_selected)
                    
        print(best_setup)
        multistrategy_best = BackTestResults()          
        for btest_id in best_setup:
            trade_hist = self.bt_results[btest_id]
            multistrategy_best.add(trade_hist, same_deposit=False)
        multistrategy_best.eval_daily_metrics()
        multistrategy_best.print_results()
        multistrategy_best.plot_results()
        multistrategy_best.save_fig()
        return multistrategy_best

    def _plot_optimization_results(self, symbol: Symbol, period) -> None:
        if self.opt_summary is None:
            self.opt_summary = self._generate_optimization_summary()

        # Individual tests results
        top_runs_ids = []
        sum_daily_profit = 0
        for symbol in set(self.opt_summary["symbol"]):
            opt_summary_for_ticker = self.opt_summary[self.opt_summary["symbol"] == symbol]
            top_runs_ids.append(opt_summary_for_ticker.index[0])
            sum_daily_profit += self.bt_results[top_runs_ids[-1]
                                            ].daily_hist["profit_csum"]
            logger.info(f"\n{opt_summary_for_ticker.head(10)}\n")
            pd.DataFrame(self.bt_results[top_runs_ids[-1]].daily_hist["profit_csum"]).to_csv(
                f"optimization/{symbol}.{period.value}.top_{self.sortby}_sorted.csv", index=False)

        profit_av = (sum_daily_profit / len(top_runs_ids)).values
        APR_av, maxwait_av = self.bt_results[top_runs_ids[-1]].metrics_from_profit(profit_av)
        logger.info(
            f"\nAverage of top runs ({', '.join([f'{i}' for i in top_runs_ids])}): APR={APR_av:.2f}, maxwait={maxwait_av:.0f}\n")
        plot_daily_balances_with_av(
            btests=self.bt_results,
            test_ids=top_runs_ids,
            profit_av=profit_av,
            metrics_av=[("APR", APR_av), ("mwait", maxwait_av)]
        )
        plt.savefig(
            f"optimization/{period.value}.av_{self.sortby}_sorted_runs.png")
        plt.clf()

        # Mix results
        opt_res = {"param_set": [], "symbol": [],
                   "final_balance": [], "ndeals": [], "test_ids": []}
        for i in range(self.opt_summary.shape[0]):
            exphash, test_ids = "", ""
            for col in self.opt_summary.columns:
                if col not in ["symbol", "APR", "ndeals", "recovery", "maxwait", "final_profit", "loss_max_rel"]:
                    exphash += str(self.opt_summary[col].iloc[i]) + " "
            opt_res["test_ids"].append(f".{self.opt_summary.index[i]}")
            opt_res["param_set"].append(exphash)
            opt_res["symbol"].append(f".{self.opt_summary['symbol'].iloc[i]}")
            opt_res["ndeals"].append(self.opt_summary.ndeals.iloc[i])
            opt_res["final_balance"].append(self.opt_summary.APR.iloc[i])

        opt_res = pd.DataFrame(opt_res)
        opt_res = opt_res.groupby(by="param_set").sum()

        recovery, maxwait, balances_av = [], [], {}
        for i in range(opt_res.shape[0]):
            sum_daily_profit = 0
            test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
            for test_id in test_ids:
                sum_daily_profit += self.bt_results[test_id].daily_hist["profit_csum"]
            profit_av = sum_daily_profit/len(test_ids)
            metrics = Metrics(self.bt_results[0].daily_hist.index, profit_av.values)
            recovery.append(metrics.recovery_factorery)
            maxwait.append(metrics.max_period)
            opt_res["final_balance"].iloc[i] = profit_av.values[-1]
            balances_av[opt_res.index[i]] = profit_av
        opt_res["recovery"] = recovery
        opt_res["maxwait"] = maxwait
        opt_res["ndeals"] = np.int64(opt_res["ndeals"].values/len(test_ids))

        opt_res.sort_values(by=[self.sortby], ascending=False, inplace=True)
        logger.info(f"\n{opt_res}\n\n")

        legend = []
        for test_id in range(min(opt_res.shape[0], 5)):
            plt.plot(self.bt_results[0].daily_hist.index, balances_av[opt_res.index[test_id]],
                     linewidth=2 if test_id == 0 else 1)
            row = opt_res.iloc[test_id]
            legend.append(
                f"{opt_res.index[test_id]}: b={row.final_balance:.0f} ({row.ndeals}) recv={row.recovery:.2f} mwait={row.maxwait:.0f}")
        plt.legend(legend)
        plt.grid("on")
        plt.tight_layout()
        plt.savefig(
            f"optimization/{period.value}.av_{self.sortby}_sorted_runs_with_same_paramset.top5.png")
        plt.clf()
        # plt.subplot(2, 1, 2)
        i = 0
        test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
        plot_daily_balances_with_av(self.bt_results,
                                    test_ids,
                                    balances_av[opt_res.index[i]].values,
                                    metrics_av=[("recovery", opt_res.iloc[i].recovery)])
        plt.tight_layout()
        plt.savefig(
            f"optimization/{period.value}.av_{self.sortby}_sorted_runs_with_same_paramset.top1.png")
        plt.clf()
