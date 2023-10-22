import sys
from easydict import EasyDict
from loguru import logger

from backtest import backtest
from experts import PyConfig

logger.remove()
logger.add(sys.stderr, level="INFO")
optim_cfg = PyConfig().optim()

import itertools
keys, values = zip(*optim_cfg.items())
cfgs = [EasyDict(zip(keys, v)) for v in itertools.product(*values)]


opt_summary = {k:[] for k in cfgs[0].keys()}
for k in opt_summary.keys():
      for cfg in cfgs:
            v = cfg[k]
            if type(v) is EasyDict and "func" in v.keys():
                  opt_summary[k].append(v.func)
            else:
                  opt_summary[k].append(v)
      opt_summary[k] = set(opt_summary[k])


logger.info("\n".join(["const params:"]+[f"{k}={v[0]}" for k, v in optim_cfg.items() if len(opt_summary[k])==1]))
for cfg in cfgs:
      logger.info("\n".join(["current params:"]+[f"{k}={v}" for k, v in cfg.items() if len(opt_summary[k])>1]))
      backtest(cfg)
        
        
"""
H4 - SBER - trngl:100 trend:200
   - GAZP - trngl:120 trend:130

H1 - SBER - trngl:140 trend:190
   - GAZP - trngl:080 trend:100    
"""

"""
trend - SBER - H4:100 H1:200
      - GAZP - H4:120 H1:130

trngl - SBER - H4:140 H1:190
      - GAZP - H4:080 H1:100    
"""

"""
trend - H4 - SBER:100 GAZP:200 -> 300
trngl - H4 - SBER:120 GAZP:130 -> 250
trend - H1 - SBER:140 GAZP:190 -> 330 +
trngl - H1 - SBER:080 GAZP:100 -> 180   
"""