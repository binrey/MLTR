import os
import pickle
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from common.type import Symbols, TimePeriod
from data_processing import PULLERS
from run import run_backtest
from trade.utils import Position

load_dotenv()

LOCAL_LOGS_DIR = Path(os.getenv("LOG_DIR"))
BROKER = "bybit"
EXPERT = "volprof"
SYMBOL = Symbols.ETHUSDT
PERIOD = TimePeriod.M1
TAG = f"{SYMBOL.ticker}-{PERIOD.value}"

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
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "START" in line:
                start_time = extract_datetimes(line)
            if "server time" in line:
                end_time = extract_datetimes(line)

    assert start_time is not None, f"No start time found in {log_file}"
    assert end_time is not None, f"No end time found in {log_file}"
    logger.info(f"Pulling data for {SYMBOL.ticker} from {start_time} to {end_time}")
    PULLERS[BROKER](SYMBOL, PERIOD, start_time, end_time)

    cfg = pickle.load(open(log_file.parent.parent / "config.pkl", "rb"))
    cfg.update({"date_start": start_time, "date_end": end_time,
                "eval_buyhold": False, "clear_logs": True, "conftype": "backtest",
                "close_last_position": False})
    btest_res = run_backtest(cfg)

    return btest_res.positions


if __name__ == "__main__":
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
        logline = f"{date_test}"
        for pos_real in positions_real[ir:]:
            date_real = pos_real.open_date.astype("datetime64[m]")
            if date_real == date_test and pos_real.side == pos_test.side:
                ir += 1
                time_lags.append((pos_real.open_date.astype("datetime64[ms]") -
                                  pos_test.open_date.astype("datetime64[ms]")).astype(int)/1000)
                match_count += 1
                slippages.append((pos_real.open_price - pos_test.open_price) * pos_test.side.value / pos_test.open_price)
                logline += f" <- OK: open slip: {slippages[-1]*100:8.4f}%, time lag: {time_lags[-1]:8.2f}s"
                break
        logger.info(logline)

    logger.info("----------------------------------------")
    logger.info(f"Mean time lag: {np.mean(time_lags):.2f}s")
    logger.info(f"Match rate: {match_count / len(positions_test) * 100:.2f}%")
    logger.info(f"Mean slippage: {np.mean(slippages)*100:.4f}%")
