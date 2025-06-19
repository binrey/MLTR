import argparse
import os
import pickle
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv
from common.utils import Logger
from loguru import logger

from backtesting.backtest_broker import TradeHistory
from backtesting.utils import BackTestResults
from common.type import ConfigType, RunType, Symbols, TimePeriod
from data_processing import PULLERS
from trade.backtest import BackTest
from trade.utils import Position

load_dotenv()

LOCAL_LOGS_DIR = Path(os.getenv("LOG_DIR"))

def parse_args():
    parser = argparse.ArgumentParser(description='Validate trading positions')
    parser.add_argument('--broker', type=str, default="bybit", help='Broker name (default: bybit)')
    parser.add_argument('--expert', type=str, default="volprof", help='Expert name (default: volprof)')
    parser.add_argument('--symbol', type=str, default="ETHUSDT",
                       choices=[attr for attr in dir(Symbols) if not attr.startswith('_')],
                       help='Trading symbol (default: ETHUSDT)')
    parser.add_argument('--period', type=str, default="M1", 
                       choices=[p.name for p in TimePeriod.__members__.values()],
                       help='Time period (default: M1)')
    return parser.parse_args()


def extract_datetimes(line):
    # Pattern to match timestamp after START
    pattern = r': (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    match = re.search(pattern, line)
    if match:
        time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        return time
    return None


def download_logs(log_dir: Path):
    if not log_dir.exists():
        remote_logs_path = os.getenv('REMOTE_LOGS')
        if not remote_logs_path:
            raise ValueError("REMOTE_LOGS environment variable is not set")

        # os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
        subprocess.run(["scp", "-r", remote_logs_path, "./"], check=True)


def process_real_log_dir(log_dir: Path):
    # Read positions from positions dir
    positions_dir = log_dir / "positions"
    positions = []
    for file in positions_dir.glob("*.json"):
        pos = Position.from_json_file(file)
        positions.append(pos)
    return positions


def process_log_dir(log_dir: Path):
    cfg_files = sorted(list((log_dir).glob("*.pkl")))
    positions_test, positions_real, cfg2process = [], [], None
    for cfg_file in cfg_files + [None]:
        if cfg_file is None:
            cfg_new = {"date_start": np.datetime64("now")}
        else:
            cfg_new = pickle.load(open(cfg_file, "rb"))
        if cfg2process is not None:
            cfg2process["date_end"] = cfg_new["date_start"]
            positions_test_, positions_real_ = process_logfile(cfg2process)
            positions_test.extend(positions_test_)
            positions_real.extend(positions_real_)

        cfg2process = cfg_new
    return positions_test, positions_real


def process_logfile(cfg: Dict[str, Any]) -> tuple[list[Position], list[Position]]:
    date_start = cfg["date_start"]
    date_end = cfg["date_end"]
    # Take into account history size
    hist_window = cfg["period"].to_timedelta()*cfg["hist_size"]
    date_start_pull = date_start - hist_window
    logger.info(
        f"Pulling data for {SYMBOL.ticker} from {date_start_pull} ({date_start} - {hist_window}) to {date_end}")
    PULLERS[BROKER](SYMBOL, PERIOD, date_start_pull, date_end)

    cfg.update({"date_start": date_start, "date_end": date_end,
                "eval_buyhold": False, "clear_logs": True, "conftype": ConfigType.BACKTEST,
                "close_last_position": False, "visualize": False, "handle_trade_errors": False})

    logger_wrapper.initialize(cfg["name"], cfg["symbol"].ticker, cfg["period"].value, cfg["clear_logs"])

    # TODO replace bybit with placeholder
    PULLERS["bybit"](**cfg)
    backtest_trading = BackTest(cfg)
    backtest_trading.initialize()
    backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)
    
    val_res = BackTestResults()
    val_res.add(backtest_trading.session)
    # bt_res = backtest_trading.postprocess()
    
    positions_real = process_real_log_dir(log_dir)
    profit_hist_real = TradeHistory(backtest_trading.session.mw, positions_real).df
    assert profit_hist_real.shape[0] > 0, "No real deals in history found"
    val_res.plot_validation()
    val_res.add_profit_curve(profit_hist_real["dates"], profit_hist_real["profit_csum_nofees"], f"{cfg['symbol'].ticker} real", "g", 3, 0.5)
    val_res.add_profit_curve(profit_hist_real["dates"], backtest_trading.session.profit_hist.df["profit_csum_nofees"], f"{cfg['symbol'].ticker} test", "r", 1, 0.5)

    val_res.save_fig()
    return backtest_trading.session.positions, positions_real


if __name__ == "__main__":
    args = parse_args()
    BROKER = args.broker
    EXPERT = args.expert
    SYMBOL = getattr(Symbols, args.symbol)
    PERIOD = getattr(TimePeriod, args.period)
    TAG = f"{SYMBOL.ticker}-{PERIOD.value}"    
    
    logger_wrapper = Logger(log_dir=os.path.join(os.getenv("LOG_DIR"), RunType.BACKTEST.value),
                            log_level=os.getenv("LOGLEVEL"))

    log_dir = LOCAL_LOGS_DIR / BROKER / EXPERT / TAG
    
    download_logs(log_dir)
    positions_test, positions_real = process_log_dir(log_dir)

    if len(positions_test) == 0:
        logger.warning(f"No positions while testing")

    positions_real.sort(key=lambda x: x.open_date)
    positions_test.sort(key=lambda x: x.open_date)

    time_lags, slippages = [], []
    match_count = 0

    logger.info("")
    logger.info(f"Validation for {len(positions_test)} positions")
    logger.info("----------------------------------------")
    ir = 0
    for pos_test in positions_test:
        date_test = pos_test.open_date.astype("datetime64[m]")
        logline = f"{date_test} {pos_test.side:<4}"
        found_real = False
        for pos_real in positions_real[ir:]:
            date_real = pos_real.open_date.astype("datetime64[m]")
            if date_real == date_test and pos_real.side == pos_test.side:
                ir += 1
                time_lags.append((pos_real.open_date.astype("datetime64[ms]") -
                                  pos_test.open_date.astype("datetime64[ms]")).astype(int)/1000)
                match_count += 1
                slippages.append((pos_real.open_price - pos_test.open_price)
                                 * pos_test.side.value / pos_test.open_price)
                logline += f" <- OK: open slip: {slippages[-1]*100:8.4f}%, time lag: {time_lags[-1]:8.2f}s"
                found_real = True
                break
            else:
                if date_real < date_test:
                    logger.info(f"{date_real.astype('datetime64[m]')} {pos_real.side:<4} -> NO TEST")
                    ir += 1
                else:
                    break
        if not found_real:
            logline += " <- NO REAL"
        logger.info(logline)

    logger.info("----------------------------------------")
    logger.info(f"Mean time lag: " + (f"{np.mean(time_lags):.2f}s" if len(positions_test) > 0 else "NO TEST"))
    logger.info(f"Match rate: " + (f"{match_count / len(positions_test) * 100:.2f}%" if len(positions_test) > 0 else "NO TEST"))
    logger.info(f"Mean slippage: " + (f"{np.mean(slippages)*100:.4f}%" if len(positions_test) > 0 else "NO TEST"))
