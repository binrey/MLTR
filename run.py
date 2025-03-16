"""Main entry point for running backtests, optimization, and live trading on Bybit."""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from backtesting.cross_validation import CrossValidation
from backtesting.optimization import Optimizer
from common.type import RunType
from common.utils import PyConfig
from trade.backtest import launch as backtest_launch
from trade.backtest import launch_multirun
from trade.bybit import launch as bybit_launch

load_dotenv() 


def run_backtest(config_path):
    # Load environment variables from .env file
    cfg = PyConfig(config_path).get_inference()
    cfg["save_backup"] = False
    backtest_launch(cfg)


def run_multirun(config_paths):
    cfgs = [PyConfig(path).get_inference() for path in config_paths]
    launch_multirun(cfgs)


def run_optimization(config_path, run_backtests):
    cfg = PyConfig(config_path).get_optimization()
    cfg["visualize"] = False
    cfg["save_backup"] = False
    cfg["save_plots"] = False
    opt = Optimizer(cfg)
    if run_backtests:
        opt.clear_cache()
        opt.run_backtests()
    results = opt.optimize(cfg)
    logger.info(f"\n{str(results)}")


def run_bybit(config_path):
    cfg = PyConfig(config_path).get_inference()
    cfg["save_backup"] = True
    cfg["save_plots"] = False
    cfg["visualize"] = False
    bybit_launch(cfg)


def run_cross_validation(config_path):
    cfg = PyConfig(config_path).get_optimization()
    cfg["visualize"] = False
    cfg["save_backup"] = False
    cfg["save_plots"] = False
    
    # Assuming data is loaded here for cross-validation
    cross_validator = CrossValidation(cfg, n_splits=6)
    results = cross_validator.cross_validate(cfg, metric="recovery")
    
    logger.info("Cross-validation results:")
    logger.info(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument(
        "run_type",
        type=str,
        choices=["backtest", "optimize", "bybit", "cross_validation", "multirun"],
        help="Type of run to execute."
    )
    parser.add_argument(
        "config_paths",
        type=str,
        nargs="+",
        help="Path to one or more configuration files. For multirun, specify multiple files."
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--run_backtests", action="store_true",
                        help="Whether to run backtests during optimization.")
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
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", rotation="10 MB")

    if run_type == RunType.BYBIT:
        run_bybit(args.config_paths[0])
    elif run_type == RunType.OPTIMIZE:
        run_optimization(args.config_paths[0], args.run_backtests)
    elif run_type == RunType.MULTIRUN:
        run_multirun(args.config_paths)
    elif run_type == RunType.BACKTEST:
        run_backtest(args.config_paths[0])
    elif run_type == RunType.CROSS_VALIDATION:
        run_cross_validation(args.config_paths[0])
    else:
        raise ValueError(f"Invalid run type: {run_type}")
