import itertools
import multiprocessing
import pickle
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from time import time
import json

import numpy as np
import pandas as pd
from easydict import EasyDict
from loguru import logger
from matplotlib import pyplot as plt

from backtest import BackTestResults, backtest
from utils import PyConfig


def plot_daily_balances_with_av(btests, test_ids, balance_av, recovery_av):
      legend = []
      for test_id in test_ids:
            btest = btests[test_id]
            plt.plot(btest.daily_balance["days"], btest.daily_balance["balance"])
            legend.append(f"{btest.cfg.ticker} balance={btest.final_balance:.0f} ({btest.ndeals}) recov={btest.metrics['recovery']:.2f} mwait={btest.metrics['maxwait']:.0f}")
      plt.plot(btests[0].daily_balance["days"], balance_av, linewidth=3, color="black")
      legend.append(f"AV balance={balance_av[-1]:.0f}, recov={recovery_av:.2f}")
      plt.legend(legend) 
      plt.grid("on")        
      plt.tight_layout()
                  
                  
class Optimizer:
      def __init__(self):
            self.data_path = None
            self.sortby = "recovery" #"recovery" #"final_balance" #

      def backtest_process(self, args):
            logger.debug(args)
            num, cfg = args
            logger.debug(f"start backtest {num}")
            locnum = 0
            while True:
                  btest = backtest(cfg)
                  if len(btest.profits) == 0:
                        break
                  # cfg.no_trading_days.update(set(pos.open_date for pos in btest.positions))
                  locnum += 1
                  pickle.dump((cfg, btest), open(str(self.data_path / f"btest.{num + locnum/100:05.2f}.{cfg.ticker}.pickle"), "wb"))
                  break


      def pool_handler(self, optim_cfg):
            if self.data_path.exists():
                  rmtree(self.data_path)
            self.data_path.mkdir(exist_ok=True)
            ncpu = multiprocessing.cpu_count()
            logger.info(f"Number of cpu : {ncpu}")

            keys, values = zip(*optim_cfg.items())
            cfgs = [EasyDict(zip(keys, v)) for v in itertools.product(*values)]
            logger.info(f"optimization steps number: {len(cfgs)}")

            # logger.info("\n".join(["const params:"]+[f"{k}={v[0]}" for k, v in optim_cfg.items() if len(param_summary[k])==1]))
            cfgs = [(i, cfg) for i, cfg in enumerate(cfgs)]
            p = Pool(ncpu)
            p.map(self.backtest_process, cfgs)
      
      
      def optimize(self, optim_cfg, run_backtests=True):
            logger.remove()
            self.data_path = Path("optimization") / f"data_{optim_cfg.period[0]}"
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
                  
            opt_summary = {k:[] for k in cfgs[0].keys()}
            opt_summary.pop("no_trading_days")
            for k in opt_summary.keys():
                  for cfg in cfgs:
                        v = cfg[k]
                        if type(v) is EasyDict and "func" in v.keys():
                              opt_summary[k].append(str(v.func.name))
                        else:
                              opt_summary[k].append(v)
            # Remove lines with attributes consists of one elment (except ticker field)                
            for k in list(opt_summary.keys()):
                  if k == "ticker":
                        continue
                  if len(set(opt_summary[k])) == 1:
                        opt_summary.pop(k)
                        
            opt_summary["final_balance"], opt_summary["ndeals"], opt_summary["recovery"], opt_summary["maxwait"] = [], [], [], []
            for btest in btests:
                  opt_summary["final_balance"].append(btest.final_balance)
                  opt_summary["ndeals"].append(btest.ndeals)
                  _, metrics = BackTestResults._calc_metrics(btest.daily_balance["balance"])
                  opt_summary["recovery"].append(metrics["recovery"])
                  opt_summary["maxwait"].append(metrics["maxwait"])
            
            opt_summary = pd.DataFrame(opt_summary)
            opt_summary.sort_values(by=["recovery"], ascending=False, inplace=True)
            
            # Individual tests results
            top_runs_ids = []
            sum_daily_balance = 0
            for ticker in set(opt_summary.ticker):
                  opt_summary_for_ticker = opt_summary[opt_summary.ticker == ticker]
                  top_runs_ids.append(opt_summary_for_ticker.index[0])
                  sum_daily_balance += btests[top_runs_ids[-1]].daily_balance["balance"]
                  logger.info(f"\n{opt_summary_for_ticker.head(10)}\n")
                  pd.DataFrame(btests[top_runs_ids[-1]].daily_balance).to_csv(f"optimization/{ticker}.{optim_cfg.period[0]}.top_{self.sortby}_sorted.csv", index=False)
                  
            balance_av = sum_daily_balance / len(top_runs_ids)
            bstair_av, metrics = BackTestResults._calc_metrics(balance_av)
            logger.info(f"\nAverage of top runs ({', '.join([f'{i}' for i in top_runs_ids])}): recovery={metrics['recovery']:.2f}, maxwait={metrics['maxwait']:.0f}\n")
            plot_daily_balances_with_av(btests=btests,
                                        test_ids=top_runs_ids, 
                                        balance_av=balance_av, 
                                        recovery_av=metrics["recovery"])
            plt.savefig(f"optimization/{optim_cfg.period[0]}.av_{self.sortby}_sorted_runs.png")
            plt.clf()
            
            # Mix results
            opt_res = {"param_set":[], "ticker":[], "final_balance":[], "ndeals":[], "test_ids":[]}
            for i in range(opt_summary.shape[0]):
                  exphash, test_ids = "", ""
                  for col in opt_summary.columns:
                        if col not in ["ticker", "final_balance", "ndeals", "recovery", "maxwait"]:
                              exphash += str(opt_summary[col].iloc[i]) + " "
                  opt_res["test_ids"].append(f".{opt_summary.index[i]}")
                  opt_res["param_set"].append(exphash)
                  opt_res["ticker"].append(f".{opt_summary.ticker.iloc[i]}")
                  opt_res["ndeals"].append(opt_summary.ndeals.iloc[i])
                  opt_res["final_balance"].append(opt_summary.final_balance.iloc[i])

            opt_res = pd.DataFrame(opt_res)
            opt_res = opt_res.groupby(by="param_set").sum()

            recovery, maxwait, balances_av = [], [], {}
            for i in range(opt_res.shape[0]):
                  sum_daily_balance = 0
                  test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
                  for test_id in test_ids:
                        sum_daily_balance += btests[test_id].daily_balance["balance"]
                  balance_av = sum_daily_balance/len(test_ids)
                  bstair_av, metrics = BackTestResults._calc_metrics(balance_av)
                  recovery.append(metrics["recovery"])
                  maxwait.append(metrics["maxwait"])
                  opt_res["final_balance"].iloc[i] = balance_av[-1]
                  balances_av[opt_res.index[i]] = balance_av
            opt_res["recovery"] = recovery
            opt_res["maxwait"] = maxwait
            opt_res["ndeals"] = np.int64(opt_res["ndeals"].values/len(test_ids))
            
            # opt_res = opt_res[opt_res.ndeals<2500]
            
            opt_res.sort_values(by=[self.sortby], ascending=False, inplace=True)
            logger.info(f"\n{opt_res}\n\n")

            # plt.figure(figsize=(8, 8))
            # plt.subplot(2, 1, 1)
            legend = []
            for test_id in range(min(opt_res.shape[0], 5)):
                  plt.plot(btests[0].daily_balance["days"], balances_av[opt_res.index[test_id]], linewidth=2 if test_id==0 else 1)
                  row = opt_res.iloc[test_id]
                  legend.append(f"{opt_res.index[test_id]}: b={row.final_balance:.0f} ({row.ndeals}) recv={row.recovery:.2f} mwait={row.maxwait:.0f}")
            plt.legend(legend) 
            plt.grid("on")  
            plt.tight_layout()
            plt.savefig(f"optimization/{optim_cfg.period[0]}.av_{self.sortby}_sorted_runs_with_same_paramset.top5.png")
            plt.clf()
            # plt.subplot(2, 1, 2)
            i = 0
            test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
            plot_daily_balances_with_av(btests, test_ids, 
                                        balances_av[opt_res.index[i]], 
                                        recovery_av=opt_res.iloc[i].recovery)
            plt.tight_layout()
            plt.savefig(f"optimization/{optim_cfg.period[0]}.av_{self.sortby}_sorted_runs_with_same_paramset.top1.png")
            plt.clf()
               
                  
if __name__ == "__main__":  
      import sys
      import argparse

      parser = argparse.ArgumentParser(description="Optimization")

      parser.add_argument("--config", type=str, help="Path to the configuration file")
      parser.add_argument("--no-backtests", action="store_true", help="Disable backtests")

      args = parser.parse_args()

      if args.config:
            print(f"Configuration file: {args.config}")
      else:
            print("No configuration file provided")

      if args.no_backtests:
            print("Backtests are disabled")
      else:
            print("Backtests are enabled")
            
      optim_cfg = PyConfig(args.config).optim()
      opt = Optimizer()
      opt.optimize(optim_cfg, run_backtests=not args.no_backtests)
