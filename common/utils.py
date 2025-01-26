import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import telebot

# from diskcache import Cache
from easydict import EasyDict
from loguru import logger
from PIL import Image

from common.type import Side

# cache = Cache(".tmp")


def name_from_cfg(cfg, name):
    return name + ":" + "-".join([f"{v}" for k, v in cfg.items()])

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
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.cfg = config_module.config
        try:
            self.optim = config_module.optimization
        except:
            pass
        
    def get_inference(self):
        cfg = deepcopy(self.cfg)
        for k, v in cfg.items():
            if type(v) is EasyDict and "func" in v.keys():
                params = EasyDict({pk: pv for pk, pv in v.params.items()})
                cfg[k].func = v.func(params)
            else:
                cfg[k] = v
        return cfg

    def get_optimization(self):
        cfg = deepcopy(self.optim)
        for k, vlist in cfg.items():
            vlist_new = []
            # If it's already a list of possible values...
            if isinstance(vlist, list):
                for v in vlist:
                    # If it’s something that has "func" and "params" (like your custom expansions)...
                    if isinstance(v, EasyDict) and "func" in v:
                        # Expand the params
                        params_list = self.unroll_params(v.params)
                        # For each params combination, make a *copy* so we don’t lose anything
                        for params in params_list:
                            new_v = deepcopy(v)
                            new_v.params = params
                            # Or if you used new_v.func = ...
                            # or any extra field that must remain, e.g. new_v.type
                            vlist_new.append(new_v)
                    else:
                        # Otherwise just keep it as is
                        vlist_new.append(v)
                cfg[k] = vlist_new
            else:
                # If not a list, wrap it in one so we have a single "variant"
                cfg[k] = [vlist]
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


import copy


def update_config(config, **kwargs):
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_config[key][sub_key] = sub_value
        else:
            new_config[key] = value
    return new_config


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

    
def date2str(date: np.datetime64, step="s") -> str:
    return np.datetime_as_string(date.astype(f"datetime64[{step}]")) if date is not None else "-/-"


class Telebot:
    def __init__(self, token) -> None:
        self.bot = telebot.TeleBot(token)
        self.chat_id = 480902846
        
    def send_image(self, img_path, caption=None):
        if img_path is None:
            return
        try:
            img = Image.open(img_path)
            if caption is not None:
                self.bot.send_message(self.chat_id, caption)
            self.bot.send_photo(self.chat_id, img)
        except Exception as ex:
            self.bot.send_message(self.chat_id, ex)

    def send_text(self, text):
        try:
            self.bot.send_message(self.chat_id, text)
        except Exception as ex:
            logger.error(f"error in sending bot message: {ex}")



def date2name(date, prefix=None):
    s = str(np.array(date).astype('datetime64[s]')).split('.')[0].replace(":", "-") 
    if prefix is not None and len(prefix) > 0:
        s += f"-{prefix}"
    return s + ".png"


