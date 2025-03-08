import itertools
import multiprocessing
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from backtesting.utils import BackTestResults
from common.type import Symbol
from common.utils import date2str
from trade.backtest import launch as backtest_launch

logger.remove()
logger.add(sys.stderr, level="INFO")
# Configure logger to filter out messages from this module
logger = logger.bind(module="data_processing.dataloading")
logger.disable("data_processing.dataloading")

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
        best_score: float  # Best score achieved during optimization
        best_config: Dict  # Configuration that produced the best results
        
        
    def __init__(self, optim_cfg: Dict[str, Any], results_dir = "results/optimization"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = Path(".cache/backtests")
        self.sortby = "APR"
        self.cfg = optim_cfg
        self.cfgs = []
        self.btests = []

    def _get_cache_subdir(self, cfg):
        """Get the appropriate cache subdirectory for a configuration."""
        if isinstance(cfg['decision_maker'], list):
            if len(cfg['decision_maker']) == 1:
                decision_maker_type = cfg['decision_maker'][0]['type'].__name__
            else:
                raise ValueError("Multiple decision makers are not supported yet")
        else:
            decision_maker_type = cfg['decision_maker']['type'].__name__
        if isinstance(cfg['symbol'], list):
            if len(cfg['symbol']) == 1:
                symbol_ticker = cfg['symbol'][0].ticker
            else:
                raise ValueError("Multiple symbols are not supported yet")
        else:
            symbol_ticker = cfg['symbol'].ticker
        return self.cache_dir / decision_maker_type / symbol_ticker

    def backtest_process(self, args):
        num, cfg = args
        logger.debug(f"start backtest {num}: {cfg}")
        locnum = 0
        while True:
            btest = backtest_launch(cfg)
            locnum += 1
            
            # Create cache subdirectory
            cache_subdir = self._get_cache_subdir(cfg)
            cache_subdir.mkdir(exist_ok=True, parents=True)
            
            # Save backtest results
            cache_file = cache_subdir / f"btest.{num + locnum/100:05.2f}.pickle"
            with open(cache_file, "wb") as f:
                pickle.dump((cfg, btest), f)
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
        return p.map(self.backtest_process, cfgs,)

    def optimize(self) -> Dict[str, OptimizationResults]:
        """
        Optimize the trading strategy for multiple symbols.
        
        Args:
            optim_cfg: Dictionary containing optimization configuration
            run_backtests: Whether to run backtests or use existing results
        
        Returns:
            Dictionary mapping symbol tickers to their OptimizationResults
        
        Raises:
            ValueError: If no symbols are specified in the configuration
        """
        if not self.cfg.get("symbol"):
            raise ValueError("No symbols specified in optimization configuration")
        
        symbols = self.cfg["symbol"]
        results = {}
        for symbol in symbols:
            logger.debug(f"Optimizing strategy for {symbol.ticker}...")
            
            # Create symbol-specific config
            symbol_cfg = deepcopy(self.cfg)
            symbol_cfg["symbol"] = [symbol]
            
            try:
                result = self._optimize_single_symbol(symbol_cfg)
                results[symbol.ticker] = result
            except Exception as e:
                logger.error(f"Failed to optimize for {symbol.ticker}: {str(e)}")
                continue
        
        return results

    def _optimize_single_symbol(self, optim_cfg) -> OptimizationResults:
        """
        Optimize the trading strategy for a specific symbol.
        
        Args:
            optim_cfg: Configuration dictionary for this symbol
            period: Trading period
            run_backtests: Whether to run backtests or use existing results
        
        Returns:
            OptimizationResults containing optimization summary and metrics
        
        Raises:
            ValueError: If configuration is invalid
        """
        assert "symbol" in optim_cfg and len(optim_cfg["symbol"]) == 1, "symbol must be a single symbol"
        symbol = optim_cfg["symbol"][0]
        # results_dir = self.results_dir / f"{symbol.value}"
        
        # Load and validate results
        self.cfgs, self.btests = self._load_backtest_results(optim_cfg)
        if not self.cfgs:
            raise ValueError(f"No backtest results found for {symbol.ticker}")
        
        # Validate dates
        self._validate_backtest_dates()
        
        # Generate optimization summary
        opt_summary = self._generate_optimization_summary()
        
        # Get best results
        top_run_id = opt_summary.index[0]
        best_score = opt_summary[self.sortby].iloc[0]
        
        # Save and plot results
        # self._plot_optimization_results(opt_summary, symbol, period)
        
        results = self.OptimizationResults(
            opt_summary=opt_summary,
            best_score=best_score,
            best_config=self.cfgs[top_run_id])
        
        # Log optimization results
        logger.info(f"Optimization results for {symbol.ticker}:")
        logger.info(f"Best score ({self.sortby}): {results.best_score:.2f}\n")  

        return results

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
            
        # Recursively find all pickle files
        for p in sorted(cache_dir.rglob("*.pickle")):
            try:
                with open(p, "rb") as f:
                    cfg, btest = pickle.load(f)
                cfgs.append(cfg)
                btests.append(btest)
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                logger.warning(f"Failed to load cache file {p}: {e}")
                continue
                
        return cfgs, btests

    def _validate_backtest_dates(self) -> None:
        """Validate that all backtests have the same start date."""
        start_dates = set([btest.date_start for btest in self.btests])
        if len(start_dates) != 1:
            raise ValueError(
                f"Inconsistent backtest start dates. Please adjust configuration to start from {date2str(max(start_dates))}"
            )

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
                        else:
                            description.append(f"{kk}:{vv}")
                    opt_summary[k].append("|".join(description))
                else:
                    if isinstance(v, Symbol):
                        v = v.ticker
                    opt_summary[k].append(v)
        
        # Add metrics
        for btest in self.btests:
            opt_summary["APR"].append(btest.APR)
            opt_summary["final_profit"].append(btest.final_profit)
            opt_summary["ndeals"].append(btest.ndeals)
            opt_summary["loss_max_rel"].append(btest.metrics["loss_max_rel"])
            opt_summary["recovery"].append(btest.metrics["recovery"])
            opt_summary["maxwait"].append(btest.metrics["maxwait"])
        
        # Convert to DataFrame and clean up
        opt_summary = pd.DataFrame(opt_summary)
        
        # Remove constant columns except symbol
        for k in list(opt_summary.columns):
            if k != "symbol" and len(set(map(str, opt_summary[k]))) == 1:
                del opt_summary[k]
        
        opt_summary.sort_values(by=["APR"], ascending=False, inplace=True)
        return opt_summary

    def _plot_optimization_results(self, opt_summary: pd.DataFrame, symbol: Symbol, period) -> None:
        # Individual tests results
        top_runs_ids = []
        sum_daily_profit = 0
        for symbol in set(opt_summary["symbol"]):
            opt_summary_for_ticker = opt_summary[opt_summary["symbol"] == symbol]
            top_runs_ids.append(opt_summary_for_ticker.index[0])
            sum_daily_profit += self.btests[top_runs_ids[-1]].daily_hist["profit_csum"]
            logger.info(f"\n{opt_summary_for_ticker.head(10)}\n")
            pd.DataFrame(self.btests[top_runs_ids[-1]].daily_hist.profit_csum).to_csv(
                f"optimization/{symbol}.{period.value}.top_{self.sortby}_sorted.csv", index=False)

        profit_av = (sum_daily_profit / len(top_runs_ids)).values
        APR_av, maxwait_av = self.btests[top_runs_ids[-1]].metrics_from_profit(profit_av)
        logger.info(
            f"\nAverage of top runs ({', '.join([f'{i}' for i in top_runs_ids])}): APR={APR_av:.2f}, maxwait={maxwait_av:.0f}\n")
        plot_daily_balances_with_av(
            btests=self.btests,
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
        for i in range(opt_summary.shape[0]):
            exphash, test_ids = "", ""
            for col in opt_summary.columns:
                if col not in ["symbol", "APR", "ndeals", "recovery", "maxwait", "final_profit", "loss_max_rel"]:
                    exphash += str(opt_summary[col].iloc[i]) + " "
            opt_res["test_ids"].append(f".{opt_summary.index[i]}")
            opt_res["param_set"].append(exphash)
            opt_res["symbol"].append(f".{opt_summary['symbol'].iloc[i]}")
            opt_res["ndeals"].append(opt_summary.ndeals.iloc[i])
            opt_res["final_balance"].append(opt_summary.APR.iloc[i])

        opt_res = pd.DataFrame(opt_res)
        opt_res = opt_res.groupby(by="param_set").sum()

        recovery, maxwait, balances_av = [], [], {}
        for i in range(opt_res.shape[0]):
            sum_daily_profit = 0
            test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
            for test_id in test_ids:
                sum_daily_profit += self.btests[test_id].daily_hist["profit_csum"]
            profit_av = sum_daily_profit/len(test_ids)
            bstair_av, metrics = BackTestResults._calc_metrics(
                profit_av.values)
            recovery.append(metrics["recovery"])
            maxwait.append(metrics["maxwait"])
            opt_res["final_balance"].iloc[i] = profit_av.values[-1]
            balances_av[opt_res.index[i]] = profit_av
        opt_res["recovery"] = recovery
        opt_res["maxwait"] = maxwait
        opt_res["ndeals"] = np.int64(opt_res["ndeals"].values/len(test_ids))

        opt_res.sort_values(by=[self.sortby], ascending=False, inplace=True)
        logger.info(f"\n{opt_res}\n\n")

        legend = []
        for test_id in range(min(opt_res.shape[0], 5)):
            plt.plot(self.btests[0].daily_hist.index, balances_av[opt_res.index[test_id]],
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
        plot_daily_balances_with_av(self.btests,
                                    test_ids,
                                    balances_av[opt_res.index[i]].values,
                                    metrics_av=[("recovery", opt_res.iloc[i].recovery)])
        plt.tight_layout()
        plt.savefig(
            f"optimization/{period.value}.av_{self.sortby}_sorted_runs_with_same_paramset.top1.png")
        plt.clf()
