from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

import pandas as pd
from loguru import logger

from common.type import Vis
from common.utils import date2name, plot_fig
from common.visualization import Visualizer
from trade.utils import Position

pd.options.mode.chained_assignment = None
import pickle
from copy import deepcopy
from multiprocessing import Process
from typing import Any

import finplot as fplt
import numpy as np
import pandas as pd
import stackprinter

from common.utils import date2str

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
        self.visualizer = Visualizer(cfg)        
        self.save_path = Path("real_trading") / f"{self.cfg.ticker}-{self.cfg.period}"
        self.backup_path = self.save_path / "backup.pkl"
        
        self.nmin = self.cfg.period.minutes
        self.time = StepData()

    @abstractmethod
    def get_server_time(self, message: Any) -> np.datetime64:
        pass

    @abstractmethod
    def update_trailing_stop(self, sl_new: float) -> None:
        pass

    @abstractmethod
    def get_current_position(self) -> Position:
        pass
    
    @abstractmethod  
    def get_hist(self):
        pass    
    
    @abstractmethod
    def get_pos_history(self) -> list[Position]:
        pass

    def handle_trade_message(self, message):
        time = self.get_server_time(message)
        time_rounded = time.astype("datetime64[m]").astype(int)//self.nmin
        self.time.update(np.datetime64(int(time_rounded*self.nmin), "m"))
        logger.info(f"server time: {date2str(time, 'ms')}")
        
        if self.time.change(no_none=True):
            msg = f"{date2str(self.time.curr)}: {str(self.pos.curr) if self.pos.curr is not None else 'no pos'}"
            self.update()
            
            logger.debug(msg)
            if self.my_telebot is not None:
                self.my_telebot.send_text(msg)
        else:
            print ("\033[A\033[A")  

    def clear_log_dir(self):
        if self.cfg.save_plots:
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

    def save_backup(self):
        backup_data = {
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

    def update_market_state(self) -> None:
        self.pos.update(self.get_current_position())

    def vis(self):
        self.visualizer(self.get_pos_history() + [self.pos.curr])

    def update(self):
        self.h = self.get_hist()
        if self.cfg.visualize != Vis.OFF:
            self.visualizer.update_hist(self.h)
        self.update_market_state()

        logger.debug(f"open: {self.h['Open'][-1]}")

        if self.pos.change(): 
            if self.cfg.visualize == Vis.ON_DEAL:
                self.vis()
            if self.my_telebot is not None:
                self.my_telebot.send_image(self.save_path / date2name(self.time))
        
        if self.cfg.visualize == Vis.ON_STEP:
            self.vis()        
        
        texp = self.exp.update(self.h, self.pos.curr)
        if self.cfg.save_backups:
            self.save_backup()


