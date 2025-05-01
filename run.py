"""Main entry point for running backtests, optimization, and live trading on Bybit."""

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from backtesting.cross_validation import CrossValidation
from backtesting.optimization import Optimizer
from common.type import RunType
from common.utils import Logger, PyConfig
from trade.backtest import launch as backtest_launch
from trade.backtest import launch_multirun
from trade.bybit import launch as bybit_launch

load_dotenv()

def run_backtest(cfg: PyConfig):
    logger_wrapper = Logger(log_dir=os.path.join(os.getenv("LOG_DIR"), RunType.BACKTEST.value),
                            log_level="DEBUG" if os.getenv("DEBUG") else "INFO")
    logger_wrapper.initialize(cfg["name"], cfg["symbol"].ticker, cfg["period"].value, True)
    cfg["save_backup"] = False
    return backtest_launch(cfg)


def run_multirun(cfgs: list[PyConfig]):
    launch_multirun(cfgs)


def run_optimization(cfg: PyConfig, run_backtests):
    cfg["visualize"] = False
    cfg["save_backup"] = False
    cfg["save_plots"] = False
    opt = Optimizer(cfg)
    if run_backtests:
        opt.clear_cache()
        opt.run_backtests()
    results = opt.optimize(cfg)
    logger.info(f"\n{str(results)}")


def run_bybit(cfg: PyConfig, demo=False):
    logger_wrapper = Logger(log_dir=os.path.join(os.getenv("LOG_DIR"), RunType.BYBIT.value),
                            log_level="DEBUG" if os.getenv("DEBUG") else "INFO")
    logger_wrapper.initialize(cfg["name"], cfg["symbol"].ticker, cfg["period"].value, True)
    cfg["save_backup"] = True
    cfg["save_plots"] = False
    cfg["visualize"] = False
    bybit_launch(cfg, demo)


def run_cross_validation(cfg: PyConfig):
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
    cfgs = [PyConfig(path) for path in args.config_paths]

    if run_type == RunType.BYBIT:
        run_bybit(cfgs[0].get_trading(), args.debug)
    elif run_type == RunType.OPTIMIZE:
        run_optimization(cfgs[0].get_optimization(), args.run_backtests)
    elif run_type == RunType.MULTIRUN:
        run_multirun([cfg.get_backtest() for cfg in cfgs])
    elif run_type == RunType.BACKTEST:
        run_backtest(cfgs[0].get_backtest())
    elif run_type == RunType.CROSS_VALIDATION:
        run_cross_validation(cfgs[0].get_optimization())
    else:
        raise ValueError(f"Invalid run type: {run_type}")
