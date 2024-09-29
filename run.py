import argparse
import sys

from loguru import logger

from backtesting.backtest import backtest
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
    cfg.save_plots = True
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

    logger.remove()
    logger.add(sys.stderr, level="INFO" if not args.debug else "DEBUG")

    if args.run_type == "bybit":
        run_bybit(args.config_path)
    elif args.run_type == "backtest":
        run_backtest(args.config_path)
    else:
        raise ValueError(f"Unknown run type: {args.run_type}")
