import json
import multiprocessing
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import stackprinter
from loguru import logger

from common.type import Vis
from common.utils import Telebot, date2str
from common.visualization import Visualizer
from experts.core.expert import ExpertBase
from trade.utils import Position

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


def log_get_hist(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, *args, **kwargs):
        result = func(self)
        if result is not None:
            logger.debug(
                f"load: {result['Date'][-1]} | new: {result['Open'][-1]}, o:{result['Open'][-2]}, h:{result['High'][-2]}, l:{result['Low'][-2]}, c:{result['Close'][-2]}")
        return result
    return wrapper


@dataclass
class StepData:
    curr: Any = None
    prev: Any = None

    def update(self, curr_value: Any):
        self.prev = deepcopy(self.curr)
        self.curr = curr_value

    def changed(self, no_none=False) -> bool:
        ch = str(self.curr) != str(self.prev)
        if no_none:
            ch = ch and self.prev is not None
        return ch

    def changed_side(self) -> bool:
        if self.curr is None or self.prev is None:
            return False
        return self.curr.side != self.prev.side

    def created(self) -> bool:
        return self.curr is not None and self.prev is None

    def deleted(self) -> bool:
        return self.curr is None and self.prev is not None


class BaseTradeClass(ABC):
    def __init__(self, cfg, expert: ExpertBase, telebot: Optional[Telebot] = None) -> None:
        self.cfg = cfg
        self.my_telebot = telebot
        self.exp = expert
        self.h, self.time = None, StepData()
        self.pos: StepData[Position] = StepData()

        self.symbol = cfg['symbol']
        self.period = cfg['period']
        self.visualize = cfg['visualize']
        self.save_plots = cfg['save_plots']
        self.vis_hist_length = cfg['vis_hist_length']
        self.vis_events = cfg['vis_events']
        self.should_save_backup = cfg['save_backup']
        self.ticker = cfg['symbol'].ticker
        self.hist_size = cfg['hist_size']
        self.log_trades = cfg['log_trades']
        self.save_path = Path(os.getenv("LOG_DIR"), cfg["conftype"], cfg["name"], f"{self.ticker}-{self.period.value}")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.save_path / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.trades_log_path = self.save_path / "positions"
        self.trades_log_path.mkdir(parents=True, exist_ok=True)

        self.visualizer = Visualizer(period=self.period,
                                     show=self.visualize,
                                     save_to=self.save_path if self.save_plots else None,
                                     vis_hist_length=self.vis_hist_length)
        self.nmin = self.period.minutes
        self.time = StepData()
        self.exp_update = self.exp.update
        self.log_config()

    @abstractmethod
    def get_server_time(self) -> np.datetime64:
        pass

    @abstractmethod
    def get_current_position(self) -> Position:
        pass

    @abstractmethod
    def get_qty_step(self) -> float:
        pass

    @abstractmethod
    def get_hist(self):
        pass

    @abstractmethod
    def get_pos_history(self) -> list[Position]:
        pass

    @abstractmethod
    def get_wallet(self) -> float:
        pass

    def get_rounded_time(self, time: np.datetime64) -> np.datetime64:
        trounded = np.array(time).astype(
            "datetime64[m]").astype(int)//self.nmin
        return np.datetime64(int(trounded*self.nmin), "m")

    def handle_trade_message(self, message):
        logger.debug("")
        server_time = self.get_server_time()
        self.time.update(self.get_rounded_time(server_time))
        if self.time.prev is None:
            logger.info(f"START: {date2str(server_time, 'ms')}")
        else:
            logger.debug(f"server time: {date2str(server_time, 'ms')}")

        if self.time.changed(no_none=True):
            self.update()
            msg = f"self.pos.curr: {str(self.pos.curr) if self.pos.curr is not None else 'None'}"
            logger.debug(msg)
            if logger._core.min_level == 10:
                print()
            if self.my_telebot is not None:
                # process = multiprocessing.Process(target=self.my_telebot.send_text,
                #                                   args=[msg])
                # process.start()
                self.my_telebot.send_text(msg)
        # else:
        #     print ("\033[A\033[A")

    def clear_log_dir(self):
        if self.save_plots:
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

    def save_backup(self):
        backup_file = self.backup_path / "backup.pkl"
        backup_data = {
            "active_position": self.pos
        }
        with open(backup_file, "wb") as f:
            pickle.dump(backup_data, f)
        logger.debug(f"backup saved to {backup_file}")

    def load_backup(self):
        backup_file = self.backup_path / "backup.pkl"
        if backup_file.exists():
            with open(backup_file, "rb") as f:
                backup_data = pickle.load(f)
            self.pos = backup_data["active_position"]
            logger.debug(f"backup loaded from {backup_file}")
        else:
            logger.debug("no backup found")

    def initialize(self):
        self.update_market_state()
        if self.pos.curr is not None:
            self.exp.active_position = self.pos.curr
            self.load_backup()
        else:
            self.clear_log_dir()
            
        logger.info(f"Market wallet: {self.get_wallet()}")

    def update_market_state(self) -> None:
        cur_pos = self.get_current_position()
        self.pos.update(cur_pos)
        logger.debug(f"update_market_state: current position: {self.pos.curr}")

    def vis(self):
        # return self.visualizer([self.pos.prev, self.pos.curr], self.exp)
        return self.visualizer(self.get_pos_history() + [self.pos.curr], self.exp)

    def log_config(self):
        config_file = self.save_path / "config.pkl"
        # with open(config_file, 'w') as f:
        #     json.dump(self.cfg, f, indent=2)
        pickle.dump(self.cfg, open(config_file, "wb"))
        logger.debug(f"Config saved to {config_file}")

    def log_trade(self, position: Position):
        """Log trade information to a separate JSON file.

        Args:
            position: The Position object containing trade details
        """
        try:
            trade_data = position.to_dict()
            open_time_str = date2str(
                position.open_date, 'ms').replace(':', '-')
            trade_file = self.trades_log_path / f"{open_time_str}.json"
            with open(trade_file, 'w') as f:
                json.dump(trade_data, f, indent=2)
            logger.debug(
                f"Logged position for {position.ticker} to {trade_file}")

        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    def update(self):
        self.h = self.get_hist()
        if self.visualize or self.save_plots:
            self.visualizer.update_hist(self.h)
        self.update_market_state()

        if self.pos.created() or self.pos.deleted() or self.pos.changed():
            if self.vis_events == Vis.ON_DEAL:
                # process = multiprocessing.Process(target=self.vis())
                # process.start()
                self.vis()
                # if self.my_telebot is not None:
                #     self.my_telebot.send_image(saved_img_path)

        if self.pos.deleted() or self.pos.changed_side():
            logger.debug(
                f"position closed {self.pos.prev.id} at {self.pos.prev.close_price}, profit: {self.pos.prev.profit_abs} ({self.pos.prev.profit}%)")
            if self.log_trades:
                self.log_trade(self.pos.prev)

        if self.vis_events == Vis.ON_STEP:
            self.vis()

        self.exp_update(self.h, self.pos.curr)
        if self.should_save_backup:
            self.save_backup()
