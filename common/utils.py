import importlib.util
import os
import shutil
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import telebot

# from diskcache import Cache
from easydict import EasyDict
from loguru import logger
from PIL import Image


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


def set_indicator_cache_dir(symbol, period, window):
    cache_dir = Path('.cache/indicators')
    expert_cache_dir = cache_dir / symbol.ticker / period.value / f"win{window}"
    expert_cache_dir.mkdir(parents=True, exist_ok=True)
    return expert_cache_dir


def init_target_from_cfg(cfg):
    cfg = deepcopy(cfg)
    Target = cfg.pop("type")
    return Target(**cfg)

def init_modules(cfg):
    for name, submodule in cfg.items():
        if not isinstance(submodule, dict):
            continue
        if "type" in submodule.keys():
            if name in ["volume_control"]:
                cfg[name] = init_target_from_cfg(submodule)
    return cfg


class PyConfig():
    def __init__(self, config_file) -> None:
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.base_config = config_module

        self.optimization, self.backetest, self.bybit = None, None, None
        if hasattr(self.base_config, "backtest"):
            self.backetest = init_modules(self.base_config.backtest)
        if hasattr(self.base_config, "bybit"):
            self.bybit = init_modules(self.base_config.bybit)
        if hasattr(self.base_config, "optimization"):
            self.optimization = init_modules(self.base_config.optimization)

    def _get_inference(self, cfg):
        cfg_compiled = deepcopy(cfg)
        for k, v in cfg_compiled.items():
            if isinstance(v, EasyDict) and "func" in v.keys():
                params = EasyDict({pk: pv for pk, pv in v.params.items()})
                cfg_compiled[k].func = v.func(params)
            else:
                cfg_compiled[k] = v
        return cfg_compiled
    
    def get_info(self) -> tuple[str, str, str]:
        for submodule in self.base_config.__dict__.values():
            if not isinstance(submodule, dict):
                continue
            if all(key in submodule for key in ["decision_maker", "symbol", "period"]):
                return (
                    submodule["decision_maker"]["type"].type,
                    submodule["symbol"].ticker,
                    submodule["period"].value
                )
        return None, None, None
    
    def get_backtest(self):
        return self._get_inference(self.backetest)
    
    def get_bybit(self):
        return self._get_inference(self.bybit)
    
    def get_optimization(self):
        cfg = deepcopy(self.optimization)
        for k, vlist in cfg.items():
            vlist_new = []
            # If it's already a list of possible values...
            if isinstance(vlist, list):
                for v in vlist:
                    # If it's something that has "func" and "params" (like your custom expansions)...
                    if isinstance(v, EasyDict) and "func" in v:
                        # Expand the params
                        params_list = self.unroll_params(v.params)
                        # For each params combination, make a *copy* so we don't lose anything
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
                cfg[k] = vlist
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
            if isinstance(v, Config):
                out += f"{k}:\n"
                for kk, vv in v.items():
                    out += f"  {kk}: {vv}\n"
            else:
                out += f"{k}: {v}\n"
        return out


def update_config(config, **kwargs):
    new_config = deepcopy(config)
    for key, value in kwargs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_config[key][sub_key] = sub_value
        else:
            new_config[key] = value
    return new_config


@dataclass

class FeeRate:
    order_execution_rate: float = field(default=0, hash=True)
    position_suply_rate: float = field(default=0, hash=True)
    order_execution_slippage_rate: float = field(default=0, hash=True)
    
    def __hash__(self):
        return hash((self.order_execution_rate, self.position_suply_rate, self.order_execution_slippage_rate))

    def order_execution_fee(self, price, volume):
        return price * volume * self.order_execution_rate / 100

    def order_execution_slippage(self, price, volume):
        return price * volume * self.order_execution_slippage_rate / 100
    
    def position_suply_fee(self, open_date, close_date, mean_price, volume):
        h8_count = np.diff([open_date, close_date]).astype('timedelta64[h]').astype(np.float32).item()/8
        return self.position_suply_rate / 100 * h8_count * mean_price * volume

    
def date2str(date: np.datetime64 | pd.Timestamp, step="s") -> str:
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y-%m-%dT%H:%M:%S")
    elif isinstance(date, np.datetime64):
        return np.datetime_as_string(date.astype(f"datetime64[{step}]"), unit=step)
    else:
        raise ValueError(f"Invalid date type: {type(date)}")


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


class Logger:
    def __init__(self, log_dir: str, log_level: str):
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        # Custom format that adds prefixes only for WARNING and ERROR
        self.format = "<level>{extra[formatted_message]}</level>"

    def initialize(self, decision_maker: str, symbol: str, period: str, clear_logs: bool):
        if clear_logs:
            if self.log_dir.exists():
                shutil.rmtree(self.log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.remove()
        
        # Custom filter to format messages with prefixes for WARNING and ERROR
        def format_message(record):
            level_name = record["level"].name
            if level_name in ["WARNING", "ERROR"]:
                record["extra"]["formatted_message"] = f"{level_name}: {record['message']}"
            else:
                record["extra"]["formatted_message"] = record["message"]
            return True
        
        logger.add(sys.stderr, level=self.log_level,
                format=self.format, filter=format_message)
        
        log_file_path = os.path.join(self.log_dir, f"{decision_maker}", f"{symbol}-{period}", "log_records", f"{datetime.now()}.log")
        logger.add(log_file_path, level=self.log_level,
                format=self.format, rotation="10 MB", filter=format_message)
