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


logger.remove()
logger.add(sys.stderr, level="INFO")

def backtest_process(args):
      logger.debug(args)
      num, cfg = args
      logger.info(f"start backtest {num}")
      locnum = 0
      while True:
            btest = backtest(cfg)
            if len(btest.positions) == 0:
                  break
            cfg.no_trading_days.update(set(pos.open_date for pos in btest.positions))
            locnum += 1
            # btest = backtest(cfg)
            btest_res = btest.profits.sum()
            logger.info(f"back test {num}: {btest_res}")
            pickle.dump((cfg, btest), open(str(Path("optimization") / f"btest.{num+locnum/100:05.2f}.pickle"), "wb"))


def pool_handler():
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
      
# for proc in procs:
#       proc.join()
      
# opt_summary = pd.DataFrame(opt_summary)
# pickle.dump(opt_summary, open(str(Path(".") / f"optim_results.pickle"), "wb"))
      
      
if __name__ == "__main__":
      save_path = Path("optimization")
      if save_path.exists():
            rmtree(save_path)
      save_path.mkdir()
      t0 = time()
      pool_handler()
      logger.info(f"optimization time: {time() - t0:.1f} sec")

      
        
        