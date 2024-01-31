import sys
from easydict import EasyDict
from loguru import logger
import pandas as pd
from backtest import backtest
from experts import PyConfig
import pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Process, Queue, Pool
import itertools
from time import time
from shutil import rmtree
from matplotlib import pyplot as plt
from backtest import BackTestResults
import numpy as np


def backtest_process(args):
      global opt_res
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
            pickle.dump((cfg, btest), open(str(Path("optimization") / "data" / f"btest.{num + locnum/100:05.2f}.{cfg.ticker}.pickle"), "wb"))
            break


def pool_handler(optim_cfg):
      save_path = Path("optimization") / "data"
      if save_path.exists():
            rmtree(save_path)
      save_path.mkdir(exist_ok=True)
      ncpu = multiprocessing.cpu_count()
      logger.info(f"Number of cpu : {ncpu}")

      keys, values = zip(*optim_cfg.items())
      cfgs = [EasyDict(zip(keys, v)) for v in itertools.product(*values)]
      logger.info(f"optimization steps number: {len(cfgs)}")

      # logger.info("\n".join(["const params:"]+[f"{k}={v[0]}" for k, v in optim_cfg.items() if len(param_summary[k])==1]))
      cfgs = [(i, cfg) for i, cfg in enumerate(cfgs)]
      p = Pool(ncpu)
      p.map(backtest_process, cfgs)
      
      
def optimize(optim_cfg, run_backtests=True):
      logger.remove()
      t0 = time()
      if run_backtests:
            pool_handler(optim_cfg)
      logger.add("optimization/opt_report.txt", level="INFO", rotation="30 seconds") 
      logger.info(f"optimization time: {time() - t0:.1f} sec\n")
            
      cfgs, btests = [], []
      for p in sorted(Path("optimization/data").glob("*.pickle")):
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
                        
      for k in list(opt_summary.keys()):
            if len(set(opt_summary[k])) == 1:
                  opt_summary.pop(k)
                  
      opt_summary["final_balance"], opt_summary["ndeals"] = [], []
      for btest in btests:
            opt_summary["final_balance"].append(btest.final_balance)
            opt_summary["ndeals"].append(btest.ndeals)
      
      opt_summary = pd.DataFrame(opt_summary)
      opt_summary.sort_values(by=["final_balance"], ascending=False, inplace=True)
      
      for ticker in set(opt_summary.ticker):
            logger.info(f"\n{opt_summary[opt_summary.ticker == ticker][opt_summary.ndeals < 2500].head(10)}\n")
      
      opt_res = {"param_set":[], "ticker":[], "final_balance":[], "ndeals":[], "test_ids":[]}
      for i in range(opt_summary.shape[0]):
            exphash, test_ids = "", ""
            for col in opt_summary.columns:
                  if col not in ["ticker", "final_balance", "ndeals"]:
                        exphash += str(opt_summary[col].iloc[i]) + " "
            opt_res["test_ids"].append(f".{opt_summary.index[i]}")
            opt_res["param_set"].append(exphash)
            opt_res["ticker"].append(f".{opt_summary.ticker.iloc[i]}")
            opt_res["ndeals"].append(opt_summary.ndeals.iloc[i])
            opt_res["final_balance"].append(opt_summary.final_balance.iloc[i])

      opt_res = pd.DataFrame(opt_res)
      opt_res = opt_res.groupby(by="param_set").sum()

      linearity, maxwait, balances_av = [], [], {}
      for i in range(opt_res.shape[0]):
            sum_daily_balance = 0
            test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
            for test_id in test_ids:
                  sum_daily_balance += btests[test_id].daily_balance
            balance_av = sum_daily_balance/len(test_ids)
            bstair_av, metrics = BackTestResults.calc_metrics(balance_av)
            linearity.append(metrics["linearity"])
            maxwait.append(metrics["maxwait"])
            opt_res["final_balance"].iloc[i] = balance_av[-1]
            balances_av[opt_res.index[i]] = balance_av
      opt_res["linearity"] = linearity
      opt_res["maxwait"] = maxwait
      opt_res["ndeals"] = np.int64(opt_res["ndeals"].values/len(test_ids))
      
      opt_res = opt_res[opt_res.ndeals<2500]
      
      sortby = "final_balance" #"linearity" #"final_balance" #
      opt_res.sort_values(by=[sortby], ascending=False, inplace=True)
      logger.info(f"\n{opt_res}\n\n")

      plt.figure(figsize=(8, 8))
      plt.subplot(2, 1, 1)
      legend = []
      for test_id in range(min(opt_res.shape[0], 5)):
            plt.plot(balances_av[opt_res.index[test_id]], linewidth=2 if test_id==0 else 1)
            row = opt_res.iloc[test_id]
            legend.append(f"{opt_res.index[test_id]}: bal={row.final_balance:.0f} ({row.ndeals}) lin={row.linearity:.2f} mwait={row.maxwait:.0f}")
      plt.legend(legend) 
      plt.grid("on")      
      plt.subplot(2, 1, 2)
      i = 0
      test_ids = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
      legend = []
      for test_id in test_ids:
            row = opt_res.iloc[i]
            _, metrics = BackTestResults.calc_metrics(btests[test_id].daily_balance)
            plt.plot(btests[test_id].daily_balance)
            legend.append(f"{btests[test_id].cfg.ticker} bal={btests[test_id].final_balance:.0f} ({btests[test_id].ndeals}) lin={metrics['linearity']:.2f} mwait={metrics['maxwait']:.0f}")
      plt.plot(balances_av[opt_res.index[i]], linewidth=3, color="black")
      plt.legend(legend) 
      plt.grid("on")        
      plt.tight_layout()
      plt.savefig(f"optimization/opt-{optim_cfg.period[0]}-{sortby}.png")
      plt.clf
               
                  
if __name__ == "__main__":  
      optim_cfg = PyConfig().optim()
      for period in ["H1", "M15"]:
            optim_cfg.period = [period]
            optimize(optim_cfg, run_backtests=True)
        
        