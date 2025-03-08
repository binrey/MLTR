"""Main entry point for running backtests, optimization, and live trading on Bybit."""

import argparse
import os
import sys
from datetime import datetime

from loguru import logger

from backtesting.optimization import Optimizer
from common.type import RunType
from common.utils import PyConfig
from trade.backtest import launch as backtest_launch
from trade.bybit import launch as bybit_launch


def run_backtest(config_path):
    cfg = PyConfig(config_path).get_inference()
    cfg["save_backup"] = False
    backtest_launch(cfg)


def run_optimization(config_path, run_backtests):
    cfg = PyConfig(config_path).get_optimization()
    cfg["visualize"] = False
    cfg["save_backup"] = False
    cfg["save_plots"] = False
    opt = Optimizer(cfg)
    if run_backtests:
        opt.run_backtests()
    results = opt.optimize()
    return results


def run_bybit(config_path):
    cfg = PyConfig(config_path).get_inference()
    cfg["save_backup"] = True
    cfg["save_plots"] = False
    cfg["visualize"] = False
    bybit_launch(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument(
        "run_type",
        type=str,
        choices=["backtest", "optimize", "bybit"],
        help="Type of run: backtest or bybit"
    )
    parser.add_argument("config_path", type=str,
                        help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--run_backtests", action="store_true",
                        help="run backtests or use previous, default true")
    args = parser.parse_args()

    run_type = RunType.from_str(args.run_type)

    LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL,
               format=(
                   "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<level>{message}</level>"
               ))
    LOG_DIR = f"logs/{run_type.value}"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_file_path = os.path.join(LOG_DIR, f"{datetime.now()}.log")
    logger.add(log_file_path, level=LOG_LEVEL,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

    if run_type == RunType.BYBIT:
        run_bybit(args.config_path)
    elif run_type == RunType.OPTIMIZE:
        run_optimization(args.config_path, args.run_backtests)
    elif run_type == RunType.BACKTEST:
        run_backtest(args.config_path)
    else:
        raise ValueError(f"Invalid run type: {run_type}")
