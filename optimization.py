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


logger.remove()
logger.add(sys.stderr, level="INFO")


def backtest_process(args):
      global opt_res
      logger.debug(args)
      num, cfg = args
      logger.info(f"start backtest {num}")
      locnum = 0
      while True:
            btest = backtest(cfg)
            if len(btest.profits) == 0:
                  break
            # cfg.no_trading_days.update(set(pos.open_date for pos in btest.positions))
            locnum += 1
            pickle.dump((cfg, btest), open(str(Path("optimization") / f"btest.{cfg.ticker}.{num + locnum/100:05.2f}.pickle"), "wb"))
            break


def pool_handler():
      opt_res = []
      ncpu = multiprocessing.cpu_count()
      logger.info(f"Number of cpu : {ncpu}")
      optim_cfg = PyConfig().optim()

      keys, values = zip(*optim_cfg.items())
      cfgs = [EasyDict(zip(keys, v)) for v in itertools.product(*values)]
      logger.info(f"optimization steps number: {len(cfgs)}")

      # logger.info("\n".join(["const params:"]+[f"{k}={v[0]}" for k, v in optim_cfg.items() if len(param_summary[k])==1]))
      cfgs = [(i, cfg) for i, cfg in enumerate(cfgs)]
      p = Pool(ncpu)
      p.map(backtest_process, cfgs)
      
      
if __name__ == "__main__":            
      save_path = Path("optimization")
      # if save_path.exists():
      #       rmtree(save_path)
      save_path.mkdir(exist_ok=True)
      t0 = time()
      # opt_res = pool_handler()
      logger.info(f"optimization time: {time() - t0:.1f} sec")
            
      cfgs, btests = [], []
      for p in sorted(Path("optimization").glob("*.pickle")):
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
      # opt_summary.sort_values(by=["final_balance"], ascending=False, inplace=True)
      
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
      
      linearity = []
      for i in range(opt_res.shape[0]):
            id1, id2 = list(map(int, opt_res.test_ids.iloc[i].split(".")[1:]))
            balance1 = btests[id1].daily_balance
            balance2 = btests[id2].daily_balance
            balance_av = (balance1+balance2)/2
            bstair_av, metrics = BackTestResults.calc_metrics(balance_av)
            linearity.append(metrics["linearity"])
      opt_res["linearity"] = linearity
      
      opt_res.sort_values(by=["final_balance"], ascending=False, inplace=True)
      print(opt_res.head(20))

      # plt.figure(figsize=(15, 7))
      # plt.plot(balance1, alpha=0.4)
      # plt.plot(balance2, alpha=0.4)
      # plt.plot(balance_av, alpha=0.4)
      # plt.plot(bstair_av)
      # plt.fill(np.hstack([balance_av - bstair_av, np.zeros(1)]))
      opt_res_id = 0
      legend = []
      
      id1, id2 = list(map(int, opt_res.test_ids.iloc[opt_res_id].split(".")[1:]))
      plt.plot(btests[id1].daily_balance, linewidth=1)
      plt.plot(btests[id2].daily_balance, linewidth=1)
      plt.plot((btests[id1].daily_balance+btests[id2].daily_balance)/2, linewidth=2)
            # legend.append(f"{test_id:03.0f} {cfgs[test_id].ticker}-{cfgs[test_id].period}: -> {btest.final_balance:.0f} ({btest.ndeals})")
      plt.plot()
      # plt.legend(legend)
      plt.grid("on")      
      plt.show()
        
        