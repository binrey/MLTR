from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import sleep

import pandas as pd
from loguru import logger

from common.type import Side
from common.utils import Telebot, date2name, plot_fig
from trade.utils import Position

pd.options.mode.chained_assignment = None
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import stackprinter
import yaml
from easydict import EasyDict
from pybit.unified_trading import HTTP, WebSocket

from common.utils import PyConfig, date2str
from experts import ByBitExpert

stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen
 


@dataclass
class StepData:
    curr: Any = None
    prev: Any = None
    
    def update(self, curr_value: Any):
        self.prev = deepcopy(self.curr)
        self.curr = curr_value
        
    def change(self, no_none=False):
        ch = str(self.curr) != str(self.prev)
        if no_none:
            ch = ch and self.prev is not None
        return ch

class BaseTradeClass(ABC):
    def __init__(self, cfg, expert, telebot) -> None:
        self.cfg = cfg
        self.my_telebot = telebot
        self.exp = expert
        self.h, self.time = None, StepData()
        self.pos: StepData[Position] = StepData()
            
        self.hist2plot, self.lines2plot, self.sl = None, None, None
        
        self.save_path = Path("real_trading") / f"{self.cfg.ticker}-{self.cfg.period}"
        self.backup_path = self.save_path / "backup.pkl"

    @abstractmethod
    def update_trailing_stop(self, sl_new: float) -> None:
        pass

    @abstractmethod
    def to_datetime(self, timestamp: Union[int, float]) -> np.datetime64:
        pass

    @abstractmethod
    def get_current_position(self) -> Position:
        pass
    
    @abstractmethod  
    def get_hist(self):
        pass    

    def clear_log_dir(self):
        if self.cfg.save_plots:
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

    def save_backup(self):
        backup_data = {
            "hist2plot": self.hist2plot,
            "lines2plot": self.lines2plot,
            "sl": self.sl,
            "open_position": self.pos
        }
        backup_path = self.save_path / "backup.pkl"
        with open(backup_path, "wb") as f:
            pickle.dump(backup_data, f)
        logger.debug(f"backup saved to {backup_path}")

    def load_backup(self):
        if self.backup_path.exists():
            with open(self.backup_path, "rb") as f:
                backup_data = pickle.load(f)
            self.hist2plot = backup_data["hist2plot"]
            self.lines2plot = backup_data["lines2plot"]
            self.sl = backup_data["sl"]
            self.pos = backup_data["open_position"]
            logger.info(f"backup loaded from {self.backup_path}")
        else:
            logger.info("no backup found")

    def test_connection(self):
        self.update_market_state()
        if self.pos.curr is not None:
            self.exp.active_position = self.pos.curr
            self.load_backup()
        else:
            self.clear_log_dir()

    def trailing_sl(self):
        if self.h is None or self.pos.curr is None:
            return
        sl = float(self.pos.curr.sl)
        sl_new = float(sl + self.cfg.trailing_stop_rate*(self.h.Open[-1] - sl))
        sl_new = sl_new if abs(sl_new - self.h.Open[-1]) - self.cfg.ticksize > 0 else sl
        self.update_trailing_stop(sl_new)
        return float(sl)       

    def update_market_state(self) -> None:
        self.pos.update(self.get_current_position())

    def update(self):
        self.update_market_state()    
                
        self.sl = self.trailing_sl()

        self.h = self.get_hist()

        if self.cfg.save_plots:
            if self.pos.change(): 
                if self.pos.prev is None:
                    self.hist2plot = pd.DataFrame(self.h)
                    self.hist2plot.set_index(pd.to_datetime(self.hist2plot.Date), drop=True, inplace=True)
                    self.lines2plot = deepcopy(self.exp.lines)
                    for line in self.lines2plot:
                        for i, point in enumerate(line):
                            y = point[1]
                            try:
                                y = y.item() #  If y is 1D numpy array
                            except:
                                pass
                            x = point[0]
                            x = max(self.hist2plot.Id.iloc[0], x)
                            x = min(self.hist2plot.Id.iloc[-1], x)
                            line[i] = (self.hist2plot.index[self.hist2plot.Id==x][0], y)    
                    open_time = pd.to_datetime(self.pos.curr.open_date.astype("datetime64[m]"))
                    self.lines2plot.append([(open_time, self.pos.curr.open_price), (open_time, self.pos.curr.open_price)])
                    self.lines2plot.append([(open_time, self.pos.curr.sl), (open_time, self.pos.curr.sl)])
                    # plot_fig(self.hist2plot, self.lines2plot, self.save_path, None, open_time, self.pos.curr.side, self.cfg.ticker))
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, open_time, self.pos.curr.side, self.cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2name(open_time))
                    self.hist2plot = self.hist2plot.iloc[:-1]                    
                else:
                    last_row = pd.DataFrame(self.h).iloc[-2:]
                    last_row.index = pd.to_datetime(last_row.Date)
                    self.hist2plot = pd.concat([self.hist2plot, last_row])   
                    open_time = pd.to_datetime(self.pos.prev.open_date.astype("datetime64[m]"))                
                    self.lines2plot[-2][-1] = (pd.to_datetime(self.hist2plot.iloc[-2].Date), self.lines2plot[-2][-1][-1])
                    self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-2].Date), self.pos.prev.sl))
                    
                    # plot_fig(self.hist2plot, self.lines2plot, self.save_path, None, open_time, side, self.cfg.ticker)
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, open_time, self.pos.prev.side, self.cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2name(open_time))
                    self.hist2plot = None    
            elif self.pos.curr is not None:
                h = pd.DataFrame(self.h).iloc[-2:-1]
                h.set_index(pd.to_datetime(h.Date), drop=True, inplace=True)
                self.hist2plot = pd.concat([self.hist2plot, h])
                if self.sl is not None:
                    self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
        
        texp = self.exp.update(self.h, self.pos.curr)
        self.save_backup()
