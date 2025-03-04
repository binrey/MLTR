import itertools
import multiprocessing
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from copy import deepcopy as copy
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from backtesting.utils import BackTestResults
from common.type import Symbol
from common.utils import PyConfig, date2str
from trade.backtest import launch as backtest_launch

logger.remove()
logger.add(sys.stderr, level="INFO")
pd.set_option('display.max_colwidth', 256)


def plot_daily_balances_with_av(btests: List[BackTestResults], test_ids: List[int], profit_av: np.ndarray, metrics_av: List[Tuple[str, float]]):
    legend = []
    for test_id in test_ids:
        btest = btests[test_id]
        plt.plot(btest.daily_hist.index, btest.daily_hist["profit_csum"])
        legend.append(f"{btest.tickers} profit={btest.final_profit:.0f} ({btest.ndeals}) APR={btest.APR:.2f} mwait={btest.metrics['maxwait']:.0f}")
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
    def __init__(self):
        self.data_path = None
        self.sortby = "recovery"

    def backtest_process(self, args):
        num, cfg = args
        logger.debug(f"start backtest {num}: {cfg}")
        locnum = 0
        while True:
            btest = backtest_launch(cfg)
            if btest.ndeals_per_month < cfg["min_deals_per_month"]:
                break
            # cfg.no_trading_days.update(set(pos.open_date for pos in btest.positions))
            locnum += 1
            pickle.dump((cfg, btest), 
                        open(str(self.data_path / f"btest.{num + locnum/100:05.2f}.{cfg['symbol'].ticker}.pickle"), "wb"))
            break


    def pool_handler(self, optim_cfg):
        if self.data_path.exists():
            rmtree(self.data_path)
        self.data_path.mkdir(exist_ok=True, parents=True)
        ncpu = multiprocessing.cpu_count()
        logger.info(f"Number of cpu : {ncpu}")

        cfgs = collect_config_combinations(optim_cfg)
        logger.info(f"optimization steps number: {len(cfgs)}")

        cfgs = [(i, cfg) for i, cfg in enumerate(cfgs)]
        p = Pool(ncpu)
        p.map(self.backtest_process, cfgs)
        # for cfg in cfgs:
        #     self.backtest_process(cfg)
    
    
    def optimize(self, optim_cfg, run_backtests=True):
        self.data_path = Path("optimization") / f"data_{optim_cfg['period'][0].value}"
        t0 = time()
        if run_backtests:
            self.pool_handler(optim_cfg)
        logger.add("optimization/opt_report.txt", level="INFO", rotation="30 seconds") 
        logger.info(f"optimization time: {time() - t0:.1f} sec\n")
            
        cfgs, btests = [], []
        for p in sorted(self.data_path.glob("*.pickle")):
            cfg, btest = pickle.load(open(p, "rb"))
            cfgs.append(cfg)
            btests.append(btest)
            print(p)
        
        assert len(btests), "Нет прогонов"
        
        start_dates = set([btest.date_start for btest in btests])
        assert len(start_dates) == 1, f"""Даты прогонов разные, скорее всего нужно сдвинуть 
                            вправо дату начала тестов в конф. 
                            файле оптимизации минимум до {date2str(max(start_dates))}"""
            
        opt_summary = defaultdict(list)
        cfg_keys = list(cfgs[0].keys())
        cfg_keys.remove("no_trading_days")
        for k in cfg_keys:
            for cfg in cfgs:
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
                
        # Remove lines with attributes consists of one elment (except symbol field)            
        for k in list(opt_summary.keys()):
            if k == "symbol":
                continue
            if len(set(map(str, opt_summary[k]))) == 1:
                opt_summary.pop(k)
                        
        for btest in btests:
            opt_summary["APR"].append(btest.APR)
            opt_summary["final_profit"].append(btest.final_profit)
            opt_summary["ndeals"].append(btest.ndeals)
            opt_summary["loss_max_rel"].append(btest.metrics["loss_max_rel"])
            opt_summary["recovery"].append(btest.metrics["recovery"])
            opt_summary["maxwait"].append(btest.metrics["maxwait"])
        
        opt_summary = pd.DataFrame(opt_summary)
        opt_summary.sort_values(by=["APR"], ascending=False, inplace=True)
        
        # Individual tests results
        top_runs_ids = []
        sum_daily_profit = 0
        for symbol in set(opt_summary["symbol"]):
            opt_summary_for_ticker = opt_summary[opt_summary["symbol"] == symbol]
            top_runs_ids.append(opt_summary_for_ticker.index[0])
            sum_daily_profit += btests[top_runs_ids[-1]].daily_hist["profit_csum"]
            logger.info(f"\n{opt_summary_for_ticker.head(10)}\n")
            pd.DataFrame(btests[top_runs_ids[-1]].daily_hist.profit_csum).to_csv(f"optimization/{symbol}.{optim_cfg['period'][0].value}.top_{self.sortby}_sorted.csv", index=False)
            
        profit_av = (sum_daily_profit / len(top_runs_ids)).values
        APR_av, maxwait_av = btests[top_runs_ids[-1]].metrics_from_profit(profit_av)
        # bstair_av, metrics = BackTestResults._calc_metrics(profit_av.values)
        logger.info(f"\nAverage of top runs ({', '.join([f'{i}' for i in top_runs_ids])}): APR={APR_av:.2f}, maxwait={maxwait_av:.0f}\n")
        plot_daily_balances_with_av(
            btests=btests,
            test_ids=top_runs_ids, 
            profit_av=profit_av, 
            metrics_av=[("APR", APR_av), ("mwait", maxwait_av)]
            )
        plt.savefig(f"optimization/{optim_cfg['period'][0].value}.av_{self.sortby}_sorted_runs.png")
        plt.clf()
        
        # Mix results
        opt_res = {"param_set":[], "symbol":[], "final_balance":[], "ndeals":[], "test_ids":[]}
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
                sum_daily_profit += btests[test_id].daily_hist["profit_csum"]
            profit_av = sum_daily_profit/len(test_ids)
            bstair_av, metrics = BackTestResults._calc_metrics(profit_av.values)
            recovery.append(metrics["recovery"])
            maxwait.append(metrics["maxwait"])
            opt_res["final_balance"].iloc[i] = profit_av.values[-1]
            balances_av[opt_res.index[i]] = profit_av
        opt_res["recovery"] = recovery
        opt_res["maxwait"] = maxwait
        opt_res["ndeals"] = np.int64(opt_res["ndeals"].values/len(test_ids))
        
        # opt_res = opt_res[opt_res.ndeals<2500]
        
        opt_res.sort_values(by=[self.sortby], ascending=False, inplace=True)
        logger.info(f"\n{opt_res}\n\n")

        legend = []
        for test_id in range(min(opt_res.shape[0], 5)):
            plt.plot(btests[0].daily_hist.index, balances_av[opt_res.index[test_id]], linewidth=2 if test_id==0 else 1)
            row = opt_res.iloc[test_id]
            legend.append(f"{opt_res.index[test_id]}: b={row.final_balance:.0f} ({row.ndeals}) recv={row.recovery:.2f} mwait={row.maxwait:.0f}")
        plt.legend(legend) 
        plt.grid("on")  
        plt.tight_layout()
        plt.savefig(f"optimization/{optim_cfg['period'][0].value}.av_{self.sortby}_sorted_runs_with_same_paramset.top5.png")
        plt.clf()
        # plt.subplot(2, 1, 2)
        i = 0
        test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
        plot_daily_balances_with_av(btests, 
                            test_ids, 
                            balances_av[opt_res.index[i]].values, 
                            metrics_av=[("recovery", opt_res.iloc[i].recovery)])
        plt.tight_layout()
        plt.savefig(f"optimization/{optim_cfg['period'][0].value}.av_{self.sortby}_sorted_runs_with_same_paramset.top1.png")
        plt.clf()

