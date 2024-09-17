import argparse

from backtesting.backtest import backtest
from common.utils import PyConfig
from trade.bybit import launch as bybit_launch


def run_backtest(config_path, debug):
    cfg = PyConfig(config_path).test()
    btest_results = backtest(cfg, loglevel="DEBUG" if debug else "INFO")
    btest_results.plot_results()

def run_bybit(config_path, debug):
    cfg = PyConfig(config_path).test()
    cfg.save_plots = True
    bybit_launch(cfg)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with specified configuration."
    )
    parser.add_argument("run_type", type=str, help="Type of run: backtest or bybit")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.run_type == "bybit":
        run_bybit(args.config_path, args.debug)
    elif args.run_type == "backtest":
        run_backtest(args.config_path, args.debug)
    else:
        raise ValueError(f"Unknown run type: {args.run_type}")
