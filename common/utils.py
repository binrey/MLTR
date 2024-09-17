import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import telebot
from diskcache import Cache
from easydict import EasyDict
from PIL import Image

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

    
def date2str(date: np.datetime64) -> str:
    return np.datetime_as_string(date.astype("datetime64[s]"))


class Telebot:
    def __init__(self, token) -> None:
        self.bot = telebot.TeleBot(token)
        self.chat_id = 480902846
    def send_image(self, img_path, caption=None):
        try:
            img = Image.open(img_path)
            if caption is not None:
                self.bot.send_message(self.chat_id, caption)
            self.bot.send_photo(self.chat_id, img)
        except Exception as ex:
            self.bot.send_message(self.chat_id, ex)

    def send_text(self, text):
            self.bot.send_message(self.chat_id, text)



def date2name(date, prefix=None):
    s = str(np.array(date).astype('datetime64[s]')).split('.')[0].replace(":", "-") 
    if prefix is not None and len(prefix) > 0:
        s += f"-{prefix}"
    return s + ".png"

def plot_fig(hist2plot, lines2plot, save_path=None, prefix=None, t=None, side=None, ticker="X"):
    lines_in_range = [[] for _ in range(len(lines2plot))]
    for i, line in enumerate(lines2plot):
        assert len(line) >= 2, "line must have more than 1 point"
        for point_id, point in enumerate(line):
            assert len(point) == 2
            point = (point[0], float(point[1]))
            assert type(point[0]) is pd.Timestamp, f"point[0]={point[0]} in line {i}, must be pd.Timestamp, but has type {type(point[0])}"
            assert type(point[1]) is float, f"point[1]={point[1]} in line {i}, must be float, but has type {type(point[1])}"
            if point[0] >= hist2plot.index[0]:
                lines_in_range[i].append(point)
            elif len(line) == 2:
                lines_in_range[i].append((hist2plot.index[-1], point[1]))

            assert point[0] <= hist2plot.index[-1]
    mystyle=mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
    kwargs = dict(
        type='candle',
        block=False,
        alines=dict(alines=lines_in_range, linewidths=[1]*len(lines2plot)),
        volume=True,
        figscale=1.5,
        style=mystyle,
        datetime_format='%m-%d %H:%M:%Y',
        title=f"{np.array(t).astype('datetime64[s]')}-{ticker}-{side}",
        returnfig=True
    )

    fig, axlist = mpf.plot(data=hist2plot, **kwargs)

    if side.lower() in ["buy", "sell"]:
        side_int = 1 if side.lower() == "buy" else -1
        x = hist2plot.index.get_loc(t)
        if type(x) is slice:
            x = x.start
        y = hist2plot.loc[t].Open
        if y.ndim > 0:
            y = y.iloc[0]
        arrow_size = (hist2plot.iloc[-10:].High - hist2plot.iloc[-10:].Low).mean()
        axlist[0].annotate("", (x, y + arrow_size*side_int), fontsize=20, xytext=(x, y),
                    color="black",
                    arrowprops=dict(
                        arrowstyle='->',
                        facecolor='b',
                        edgecolor='b'))

    if save_path is not None:
        save_path = save_path / date2name(t, prefix)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close('all')
    return save_path