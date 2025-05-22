import argparse
import os
import pickle
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from common.type import ConfigType, Symbols, TimePeriod
from data_processing import PULLERS
from run import run_backtest
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

    # Read positions from positions dir
    positions_dir = log_dir / "positions"
    positions = []
    for file in positions_dir.glob("*.json"):
        positions.append(Position.from_json_file(file))
    return positions


def process_log_dir(log_dir):
    log_files = sorted(list((log_dir / "log_records").glob("*.log")))
    positions = []
    for log_file in log_files:
        positions.extend(process_logfile(log_file))
    return positions


def process_logfile(log_file) -> list[Position]:
    start_time, end_time = None, None
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if "START" in line:
                start_time = np.datetime64(extract_datetimes(line), "[m]")
            if "server time" in line:
                end_time = np.datetime64(extract_datetimes(line), "[m]")

    assert start_time is not None, f"No start time found in {log_file}"
    assert end_time is not None, f"No end time found in {log_file}"
    
    cfg = pickle.load(open(log_file.parent.parent / "config.pkl", "rb"))
    # Take into account history size
    hist_window = cfg["period"].to_timedelta()*cfg["hist_size"]
    date_start_pull = start_time - hist_window
    logger.info(
        f"Pulling data for {SYMBOL.ticker} from {date_start_pull} ({start_time} - {hist_window}) to {end_time}")
    PULLERS[BROKER](SYMBOL, PERIOD, date_start_pull, end_time)

    cfg.update({"date_start": start_time, "date_end": end_time,
                "eval_buyhold": False, "clear_logs": True, "conftype": ConfigType.BACKTEST,
                "close_last_position": False, "visualize": False, "handle_trade_errors": False})
    btest_res = run_backtest(cfg)

    return btest_res.positions


if __name__ == "__main__":
    args = parse_args()
    BROKER = args.broker
    EXPERT = args.expert
    SYMBOL = getattr(Symbols, args.symbol)
    PERIOD = getattr(TimePeriod, args.period)
    TAG = f"{SYMBOL.ticker}-{PERIOD.value}"
    
    log_dir = LOCAL_LOGS_DIR / BROKER / EXPERT / TAG
    positions_real = download_logs(log_dir)
    positions_test = process_log_dir(log_dir)

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
