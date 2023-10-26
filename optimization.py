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
logger.info(f"optimization steps number: {len(cfgs)}")

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
opt_summary = {k:[] for k, v in optim_cfg.items() if len(param_summary[k])>1}
opt_summary["btest"] = []
# for k in param_summary.keys():
#       v = cfg[k]
#       if type(v) is EasyDict and "func" in v.keys():
#             params = {f"{v.name}.{k}" for k, v in v.items()}
#             opt_summary.update(params)
#       else:
#             opt_summary.update({k:v})
for cfg in cfgs:
      logger.info("\n".join(["current params:"]+[f"{k}={v}" for k, v in cfg.items() if len(param_summary[k])>1]))
      for k, v in opt_summary.items():
            if k in cfg.keys():
                  v.append(cfg[k])
      btest = backtest(cfg)
      btest_res = btest.profits.sum()
      opt_summary["btest"].append(btest_res)
      logger.info(f"back test: {btest}")
      
opt_summary = pd.DataFrame(opt_summary)
print(opt_summary)
      
      
      
        
        