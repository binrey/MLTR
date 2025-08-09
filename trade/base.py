import json
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import stackprinter
from loguru import logger

from common.type import Side, Symbol, Vis
from common.utils import Telebot, date2str
from common.visualization import Visualizer
from experts.core.expert import Expert
from trade.utils import Position

pd.options.mode.chained_assignment = None


stackprinter.set_excepthook(style='color')
# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen


@dataclass
class StepData:
    curr: Position | np.datetime64 | None = None
    prev: Position | np.datetime64 | None = None

    def update(self, curr_value: Position | np.datetime64):
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


class ConfigLogger:
    """
    A class to handle configuration logging operations including saving, loading, and updating configs.
    """
    
    def __init__(self, save_path: Path, date_start: Optional[np.datetime64] = None, cfg: Optional[dict[str, Any]] = None):
        """
        Initialize ConfigLogger with the save path and optional start date.
        
        Args:
            save_path: Path where config files will be saved
            date_start: Start date for the config filename. If None, will be set when log_config is called
        """
        self.config_path = save_path / f"config_{date2str(date_start, 'm')}.pkl"
        self.cfg = cfg if cfg is not None else {}
    
    def log_config(self) -> None:
        """
        Log configuration to a pickle file. If a config file already exists, it will be loaded,
        updated with the new config (if provided), and saved back.
        
        Args:
            cfg: Configuration dictionary to save/update. If None, only loads existing config
            date_start: Start date for the config filename. If None, uses self.date_start or current time
        """
        with open(self.config_path, 'wb') as f:
            pickle.dump(self.cfg, f)
            logger.debug(f"Config saved at {self.config_path}")
    
    def load_config(self) -> Optional[dict[str, Any]]:
        """
        Load configuration from a pickle file.
        
        Args:
            date_start: Start date for the config filename. If None, uses self.date_start
            
        Returns:
            Configuration dictionary if found, None otherwise
        """
        try:
            with open(self.config_path, 'rb') as f:
                config = pickle.load(f)
                logger.debug(f"Config loaded from {self.config_path}")
                return config
        except (FileNotFoundError, EOFError):
            logger.debug(f"No config found at {self.config_path}")
            return None
    
    def update_config(self, time: np.datetime64) -> None:
        """
        Update an existing configuration file with new values.
        
        Args:
            cfg: Configuration dictionary to update with
            date_start: Start date for the config filename. If None, uses self.date_start
        """
        if "date_start" not in self.cfg:
            self.cfg.update({"date_start": time})
            logger.debug(f"Update config date_start: {self.cfg['date_start']}")        
        self.cfg.update({"date_end": time})
        logger.debug(f"Update config date_end: {self.cfg['date_end']}")


class BaseTradeClass(ABC):
    def __init__(self, cfg: dict[str, Any], expert: Expert, telebot: Optional[Telebot] = None) -> None:

        self.my_telebot = telebot
        self.exp = expert
        self.h, self.time = None, StepData()
        self.pos: StepData[Position] = StepData()

        self.runtime_type = cfg["conftype"]
        self.ticker = cfg["symbol"].ticker
        self.qty_step = cfg["symbol"].qty_step
        self.tick_size = cfg["symbol"].tick_size
        self.stops_step = cfg["symbol"].stops_step
        self.period = cfg['period']
        self.visualize = cfg['visualize']
        self.save_plots = cfg['save_plots']
        self.vis_hist_length = cfg['vis_hist_length']
        self.vis_events = cfg['vis_events']
        self.should_save_backup = cfg['save_backup']
        self.hist_size = cfg['hist_size']
        self.log_trades = cfg['log_trades']
        self.handle_trade_errors = cfg['handle_trade_errors']
        self.save_path = Path(os.getenv(
            "LOG_DIR"), cfg["conftype"].value, cfg["name"], f"{self.ticker}-{self.period.value}")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.save_path / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.trades_log_path = self.save_path / "positions"
        self.trades_log_path.mkdir(parents=True, exist_ok=True)
        self.traid_stops_min_size = cfg["traid_stops_min_size"]
        self.date_start = np.datetime64(datetime.now())

        assert self.qty_step == self.get_qty_step()

        self.visualizer = Visualizer(period=self.period,
                                     show=self.visualize,
                                     save_to=self.save_path if self.save_plots else None,
                                     vis_hist_length=self.vis_hist_length)
        self.nmin = self.period.minutes
        self.time = StepData()
        self.exp_update = self.exp.update
        
        # Initialize ConfigLogger
        self.config_logger = ConfigLogger(self.save_path, self.date_start, cfg)
        self.config_logger.log_config()

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

    def __get_hist(self):
        result = self.get_hist()
        logger.debug(
            f"get data: {result['Date'][-1]} | new: {result['Open'][-1]}, o:{result['Open'][-2]}, h:{result['High'][-2]}, l:{result['Low'][-2]}, c:{result['Close'][-2]}, v:{result['Volume'][-2]}")
        return result

    @abstractmethod
    def get_pos_history(self) -> list[Position]:
        pass

    @abstractmethod
    def get_deposit(self) -> float:
        pass

    @abstractmethod
    def _create_orders(self, side: Side, volume: float, time_id: Optional[int] = None):
        pass

    @abstractmethod
    def _modify_sl(self, sl: Optional[float]):
        pass

    def modify_sl(self, sl: Optional[float], price: float):
        if sl is None or self.pos.curr is None:
            return
        sl = Symbol.round_stops(self.stops_step, sl)

        if self.pos.curr.side is Side.BUY:
            min_sl = Symbol.round_stops(self.stops_step, price * (1 - self.traid_stops_min_size/100))
            if sl > min_sl:
                logger.warning(f"SL is too high: {sl} > {min_sl}, setting to {min_sl}")
                sl = min_sl
        else:
            max_sl = Symbol.round_stops(self.stops_step, price * (1 + self.traid_stops_min_size/100))
            if sl < max_sl:
                logger.warning(f"SL is too low: {sl} < {max_sl}, setting to {max_sl}")
                sl = max_sl

        self._modify_sl(sl)
        open_position: Position = self.get_open_position()
        if open_position is None:
            return

        sl_from_broker, n_attempts = open_position.sl, 0
        while sl_from_broker != sl:
            sleep(1)
            sl_from_broker = self.get_open_position().sl
            n_attempts += 1
            if n_attempts > 10:
                logger.error(f"Failed to verify SL after {n_attempts} attempts: {sl_from_broker} -> {sl}")
                break
        self.pos.curr.update_sl(sl_from_broker, self.time.curr)

    @abstractmethod
    def _modify_tp(self, tp: Optional[float]):
        pass

    def modify_tp(self, tp: Optional[float]):
        self._modify_tp(tp)

    @abstractmethod
    def get_open_position(self):
        pass

    def _compute_volume_diff(self, open_position: Position, target_volume: float):
        cur_volume, cur_side = 0, Side.NONE
        if open_position is not None:
            cur_volume = open_position.volume
            cur_side = open_position.side
        return Symbol.round_qty(qty=target_volume - cur_volume*cur_side.value, qty_step=self.qty_step)

    def create_orders(self, side: Side, volume: float, time_id: Optional[int] = None):
        volume = Symbol.round_qty(qty=volume, qty_step=self.qty_step)
        if not volume:
            return
        if self.pos.curr is None:
            target_volume = volume*side.value
        else:
            target_volume = Symbol.round_qty(qty=self.pos.curr.volume*self.pos.curr.side.value + volume*side.value, qty_step=self.qty_step)
        self._create_orders(side, volume, time_id)

        if self.handle_trade_errors:
            open_position = self.get_open_position()
            if open_position is None and target_volume != 0:
                sleep(2)
                logger.warning(f"Get none open position, while target volume is {target_volume} - double check for open position...")
                open_position = self.get_open_position()
        
            volume_diff = self._compute_volume_diff(self.get_open_position(), target_volume)
            n_attempts = 0
            while volume_diff != 0:
                logger.warning(f"Volume diff: {volume_diff}, target_volume: {target_volume}")
                self._create_orders(side=Side.from_int(volume_diff), volume=abs(volume_diff), time_id=time_id)
                volume_diff = self._compute_volume_diff(self.get_open_position(), target_volume)
                n_attempts += 1
                if n_attempts > 10:
                    logger.error(f"Failed to create orders after {n_attempts} attempts: {volume_diff} -> {target_volume}")
                    break

    def get_rounded_time(self, time: np.datetime64) -> np.datetime64:
        trounded = np.array(time).astype(
            "datetime64[m]").astype(int)//self.nmin
        return np.datetime64(int(trounded*self.nmin), "m")

    def _handle_trade_message(self):
        server_time = self.get_server_time()
        self.time.update(self.get_rounded_time(server_time))
        logger.debug(f"\n> {date2str(server_time, 'ms')} -----------------------------")

        if self.time.changed(no_none=True):
            self.update()
            self.config_logger.update_config(self.time.curr)
            self.config_logger.log_config()
            msg = f"{self.ticker}-{self.period.value}: {str(self.pos.curr) if self.pos.curr is not None else 'None'}"
            logger.debug(msg)
            if self.my_telebot is not None:
                # process = multiprocessing.Process(target=self.my_telebot.send_text,
                #                                   args=[msg])
                # process.start()
                self.my_telebot.send_text(msg)

    def handle_trade_message(self):
        self._handle_trade_message()
        for n_attempt in range(5):
            logger.debug(f"Check data: attempt {n_attempt + 1}/5, wait 10 sec...")
            sleep(10)
            h4test = self.__get_hist()
            if self.h is not None and h4test["Volume"][-2] != self.h["Volume"][-2]:
                logger.warning(f"Volume mismatch (new:{h4test['Volume'][-2]} != old:{self.h['Volume'][-2]}) after {n_attempt + 1} attempts, reevaluate...")
                self.h = h4test
                self.update(get_history=False)
            else:
                logger.debug("Got valid data, OK!")
                break

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

    def initialize(self, close_current_position: bool = False):
        self.update_market_state()
        if self.pos.curr is not None:
            self.exp.active_position = self.pos.curr
            if close_current_position:
                self.exp.close_current_position()
                sleep(3)
                self.update_market_state()
                self.clear_log_dir()
            else:
                self.load_backup()

        logger.info(f"Market wallet: {self.get_deposit()}")

    def update_market_state(self) -> None:
        self.deposit = self.get_deposit()
        self.pos.update(self.get_current_position())
        logger.debug(f"update_market_state: {self.pos.curr}")

    def vis(self):
        # return self.visualizer([self.pos.prev, self.pos.curr], self.exp)
        return self.visualizer(self.get_pos_history() + [self.pos.curr], self.exp)

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

    def update(self, get_history=True):
        if get_history:
            self.h = self.__get_hist()
        if self.visualize or self.save_plots:
            self.visualizer.update_hist(self.h)
        self.update_market_state()

        if self.pos.created() or self.pos.deleted() or self.pos.changed():
            if self.vis_events == Vis.ON_DEAL:
                self.vis()

        if self.pos.deleted() or self.pos.changed_side():
            if self.log_trades:
                self.log_trade(self.pos.prev)

        if self.vis_events == Vis.ON_STEP:
            self.vis()

        self.exp_update(self.h, self.pos.curr, self.deposit)
        if self.should_save_backup:
            self.save_backup()