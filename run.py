import argparse

from backtesting.backtest import backtest
from utils import PyConfig


def run_backtest(config_path):
    cfg = PyConfig(config_path).test()
    btest_results = backtest(cfg)
    btest_results.plot_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    run_backtest(args.config_path)
