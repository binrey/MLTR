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
from loguru import logger

from backtesting.backtest_broker import TradeHistory
from backtesting.utils import BackTestResults
from common.type import ConfigType, RunType, Symbols, TimePeriod
from common.utils import Logger
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


def parse_market_wallet_from_logs(log_dir: Path) -> float:
    """Extract market wallet value from the most recent log file."""
    log_records_dir = log_dir / "log_records"
    assert log_records_dir.exists(), f"Log records directory not found: {log_records_dir}"
    
    # Get the most recent log file
    log_files = sorted(log_records_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not log_files:
        return None
        
    # Read the most recent log file
    with open(log_files[0], 'r') as f:
        for line in f:
            if "Market wallet:" in line:
                try:
                    return float(line.split("Market wallet:")[-1].strip())
                except (ValueError, IndexError):
                    continue
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
    for file in sorted(positions_dir.glob("*.json")):
        pos = Position.from_json_file(file)
        positions.append(pos)
        
    # Bybit unexpectedly missmatch current close date and price with previous position
    for ir in range(1, len(positions)):
        if positions[ir].close_date == positions[ir-1].close_date:
            pos = positions[ir]
            close_date, close_indx, close_price = None, None, None
            if pos.sl_hist is not None and len(pos.sl_hist) > 0:
                close_date = pos.sl_hist[-1][0] + PERIOD.to_timedelta()
                close_indx = pos.close_indx
                close_price = pos.sl_hist[-1][1]
            pos.close(close_price, close_date, close_indx)
    return positions


def process_log_dir(log_dir: Path) -> tuple[list[Position], list[Position]]:
    cfg_files = sorted(list((log_dir).glob("*.pkl")))
    positions_real = process_real_log_dir(log_dir)

    positions_test, cfg2process = [], None
    for cfg_file in cfg_files + [None]:
        if cfg_file is None:
            cfg_new = {"date_start": np.datetime64("now")}
        else:
            # Start of the new cfg
            cfg_new = pickle.load(open(cfg_file, "rb"))
        if cfg2process is not None:
            positions_test.extend(process_logfile(cfg2process, positions_real))

        cfg2process = cfg_new
    
    return positions_test, positions_real


def process_logfile(cfg: Dict[str, Any], positions_real: list[Position]) -> tuple[list[Position]]:
    date_start = cfg["date_start"]
    date_end = cfg["date_end"]
    # Take into account history size
    hist_window = cfg["period"].to_timedelta()*cfg["hist_size"]
    date_start_pull = date_start - hist_window
    logger.info(
        f"Pulling data for {SYMBOL.ticker} from {date_start_pull} ({date_start} - {hist_window}) to {date_end}")
    PULLERS[BROKER](SYMBOL, PERIOD, date_start_pull, date_end)

    log_dir = Path(os.getenv("LOG_DIR")) / BROKER / EXPERT / f"{SYMBOL.ticker}-{PERIOD.value}"
    
    # Try to get market wallet from logs first
    market_wallet = parse_market_wallet_from_logs(log_dir)
    if market_wallet is not None:
        logger.info(f"Loaded market wallet from logs: {market_wallet}")

    cfg.update({
        "date_start": date_start,
        "date_end": date_end,
        "eval_buyhold": False,
        "clear_logs": True,
        "conftype": ConfigType.BACKTEST,
        "close_last_position": False,
        "visualize": False,
        "save_plots": False,
        "handle_trade_errors": False,
        # "wallet": market_wallet,
        "vis_hist_length": 1024,
    })

    logger_wrapper.initialize(cfg["name"], cfg["symbol"].ticker, cfg["period"].value, cfg["clear_logs"])

    # TODO replace bybit with placeholder
    PULLERS["bybit"](**cfg)
    backtest_trading = BackTest(cfg)
    backtest_trading.initialize()
    backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)

    val_res = BackTestResults()
    val_res.add(backtest_trading.session.profit_hist)

    profit_hist_real = TradeHistory(backtest_trading.session.mw, positions_real).df
    assert profit_hist_real.shape[0] > 0, "No real deals in history found"
    val_res.plot_validation(y_label="Fin result")
    val_res.add_profit_curve(profit_hist_real["dates"], profit_hist_real["profit_csum_nofees"], f"{cfg['symbol'].ticker} real", "g", 3, 0.5)
    val_res.add_profit_curve(profit_hist_real["dates"], backtest_trading.session.profit_hist.df["profit_csum_nofees"], f"{cfg['symbol'].ticker} test", "r", 1, 0.5)

    val_res.save_fig()
    val_res.show_fig()
    return backtest_trading.session.positions


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

    time_lags, open_slippages, close_slippages = [], [], []
    match_count, prof_diff_summ = 0, 0

    logger.info("")
    logger.info(f"Validation for {len(positions_test)} positions")
    logger.info("----------------------------------------")
    logger.info(f"{'Open Date':<16} {'Side':<11} {'Open Slip,%':>15} {'Time Lag,s':>15} {'Prof.diff,$':>15} {'Close Slip,%':>15}")
    ir, sum_prof_real, sum_prof_test = 0, 0, 0
    for pos_test in positions_test:
        date_test = pos_test.open_date.astype("datetime64[m]")
        logline, logline_suffix = f"{date_test} {pos_test.side:<4}", ""
        found_real = False
        for pos_real in positions_real[ir:]:
            date_real = pos_real.open_date.astype("datetime64[m]")
            if date_real == date_test and pos_real.side == pos_test.side:
                ir += 1
                time_lags.append((pos_real.open_date.astype("datetime64[ms]") -
                                  pos_test.open_date.astype("datetime64[ms]")).astype(int)/1000)
                match_count += 1
                
                # Calculate open slippage
                open_slip = (pos_test.open_price - pos_real.open_price) * pos_test.side.value * pos_test.volume
                open_slippages.append(open_slip)
                prof_diff = pos_test.profit_abs - pos_real.profit_abs
                prof_diff_summ += prof_diff
                logline += f" <- OK: {open_slip:15.2f} {time_lags[-1]:15.2f} {prof_diff:15.2f}"
                found_real = True

                # Bybit unexpectedly missmatch current close date and price with previous position
                if pos_real.close_date.astype("datetime64[m]") != pos_test.close_date:
                    # if pos_real.close_date == positions_real[ir-1].close_date:
                    #     pos_real.close_date = pos_test.close_date
                    #     pos_real.close_price = pos_test.close_price
                    #     if pos_real.sl_hist is not None and len(pos_real.sl_hist) > 0:
                    #         pos_real.close_price = pos_real.sl_hist[-1][1]
                    #     logline_suffix = " (REPAIRED)"
                    # else:
                    logline += f" CLOSE ERROR: real:{pos_real.close_date} test:{pos_test.close_date}"
                
                if pos_real.close_date.astype("datetime64[m]") == pos_test.close_date:
                    close_slip = -(pos_test.close_price - pos_real.close_price) * pos_test.side.value * pos_test.volume
                    close_slippages.append(close_slip)
                    logline += f" {close_slip:15.2f}"
                break

            else:
                if date_real < date_test:
                    logger.info(f"{date_real.astype('datetime64[m]')} {pos_real.side:<4} -> NO TEST")
                    ir += 1
                else:
                    break
        if not found_real:
            logline += " <- NO REAL"
        logger.info(logline + logline_suffix)

    logger.info(" "*29 + f"{np.sum(open_slippages):15.2f} {np.mean(time_lags):15.2f} {prof_diff_summ:15.2f} {np.sum(close_slippages):15.2f}")
    logger.info("----------------------------------------")
    logger.info(f"Mean time lag:       " + (f"{np.mean(time_lags):.2f}s" if len(positions_test) > 0 else "NO TEST"))
    logger.info(f"Match rate:          " + (f"{match_count / len(positions_test) * 100:.2f}%" if len(positions_test) > 0 else "NO TEST"))
    logger.info(f"Mean open slippage:  " + (f"{np.mean(open_slippages)*100:.4f}%" if len(open_slippages) > 0 else "NO TEST"))
    logger.info(f"Mean close slippage: " + (f"{np.mean(close_slippages)*100:.4f}%" if len(close_slippages) > 0 else "NO TEST"))
