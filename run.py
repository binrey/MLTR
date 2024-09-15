import argparse

from backtesting.backtest import backtest
from utils import PyConfig


def run_backtest(config_path, debug):
    cfg = PyConfig(config_path).test()
    btest_results = backtest(cfg, loglevel="DEBUG" if debug else "INFO")
    btest_results.plot_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    run_backtest(args.config_path, args.debug)
