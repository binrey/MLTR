import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from diskcache import Cache
from easydict import EasyDict

from type import PosSide

cache = Cache(".tmp")

def cache_result(func):
    def wrapper(*args, **kwargs):
        key = (func.__name__, args, frozenset(kwargs.items()))
        if key in cache:
            print("Using cached result")
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper

class PyConfig():
    def __init__(self, config_file) -> None:
        # Create a spec object
        spec = importlib.util.spec_from_file_location("config", f"configs/{config_file}")
        # Load the module from the spec
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.cfg = config_module.config
        
    def test(self):
        cfg = deepcopy(self.cfg)
        for k, v in cfg.items():
            v = v.test
            if type(v) is EasyDict and "func" in v.keys():
                params = EasyDict({pk: pv.test for pk, pv in v.params.items()})
                cfg[k].func = v.func(params)
            else:
                cfg[k] = v
        return cfg

    def optim(self):
        cfg = deepcopy(self.cfg)
        for k, vlist in cfg.items():
            vlist_new = []
            for v in vlist.optim:
                if type(v) is EasyDict and "func" in v.keys():
                    v.params = {pk: pv.optim for pk, pv in v.params.items()}
                    # v.func = partial(v.func, cfg=params)
                    params_list = self.unroll_params(v.params)
                    vlist_new += [EasyDict(func=v.func(params)) for params in params_list]
                else:
                    vlist_new.append(v)
            cfg[k] = vlist_new
        return cfg
    
    @staticmethod
    def unroll_params(cfg):
        import itertools
        keys, values = zip(*cfg.items())
        return [EasyDict(zip(keys, v)) for v in itertools.product(*values)]


class Config(EasyDict):
    def __str__(self):
        out = "config file:\n"
        for k, v in self.__dict__.items():
            if type(v) is Config:
                out += f"{k}:\n"
                for kk, vv in v.items():
                    out += f"  {kk}: {vv}\n"
            else:
                out += f"{k}: {v}\n"
        return out
    
    

@dataclass
class FeeModel(ABC):
    order_execution_rate: float = field(default=0, hash=True)
    position_suply_rate: float = field(default=0, hash=True)
    
    @abstractmethod
    def order_execution_fee(self, price, volume):
        pass
    
    @abstractmethod
    def position_suply_fee(self, open_date, close_date, mean_price, volume):
        pass
    
    def __hash__(self):
        return hash((self.order_execution_rate, self.position_suply_rate))

class FeeRate(FeeModel):
    def order_execution_fee(self, price, volume):
        return price * volume * self.order_execution_rate / 100
    
    def position_suply_fee(self, open_date, close_date, mean_price, volume):
        h8_count = np.diff([open_date, close_date]).astype('timedelta64[h]').astype(np.float32).item()/8
        return self.position_suply_rate / 100 * h8_count * mean_price * volume
    
class FeeConst(FeeModel):
    def order_execution_fee(self, price, volume):
        return volume * self.order_execution_rate
    
    def position_suply_fee(self, open_date, close_date, mean_price, volume):
        h8_count = np.diff([open_date, close_date]).astype('timedelta64[h]').astype(np.float32).item()/24
        return self.position_suply_rate * h8_count * volume

    
def side_from_str(side: str):
    if side.lower() == "buy":
        return PosSide.buy
    elif side.lower() == "sell":
        return PosSide.sell
    else:
        raise ValueError(f"{side} is not valid value, set ['buy' or 'sell']")