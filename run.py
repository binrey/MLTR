import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from loguru import logger

from common.type import RunType
from common.utils import PyConfig
from trade.backtest import launch as backtest_launch
from trade.bybit import launch as bybit_launch


def run_backtest(config_path):
    cfg = PyConfig(config_path).test()
    cfg.save_backup = False
    backtest_launch(cfg)
    

def run_bybit(config_path):
    cfg = PyConfig(config_path).test()
    cfg.save_backup = True
    cfg.save_plots = False
    cfg.visualize = False
    bybit_launch(cfg)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument("run_type", type=str, help="Type of run: backtest or bybit")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    run_type = RunType.from_str(args.run_type)

    log_level = "INFO" if not args.debug else "DEBUG"
    logger.remove()
    logger.add(sys.stderr, log_level)
    log_dir = f"logs/{run_type.value}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file_path = os.path.join(log_dir, f"{datetime.now()}.log")
    logger.add(log_file_path, level=log_level)

    if run_type == RunType.BYBIT:
        run_bybit(args.config_path)
    elif run_type == RunType.BACKTEST:
        run_backtest(args.config_path)
