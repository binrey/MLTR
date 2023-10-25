import sys
from easydict import EasyDict
from loguru import logger
import pandas as pd
from backtest import backtest
from experts import PyConfig

logger.remove()
logger.add(sys.stderr, level="INFO")
optim_cfg = PyConfig().optim()

import itertools
keys, values = zip(*optim_cfg.items())
cfgs = [EasyDict(zip(keys, v)) for v in itertools.product(*values)]


param_summary = {k:[] for k in cfgs[0].keys()}
for k in param_summary.keys():
      for cfg in cfgs:
            v = cfg[k]
            if type(v) is EasyDict and "func" in v.keys():
                  param_summary[k].append(v.func)
            else:
                  param_summary[k].append(v)
      param_summary[k] = set(param_summary[k])


logger.info("\n".join(["const params:"]+[f"{k}={v[0]}" for k, v in optim_cfg.items() if len(param_summary[k])==1]))
opt_summary = {k:[] for k in param_summary.keys()}
for cfg in cfgs:
      logger.info("\n".join(["current params:"]+[f"{k}={v}" for k, v in cfg.items() if len(param_summary[k])>1]))
      for k, v in opt_summary.items():
            if type(cfg[k]) is EasyDict and "func" in cfg[k].keys():
                  v.append(cfg[k].func.keywords["cfg"])
      btest = backtest(cfg)
      
      
        
        