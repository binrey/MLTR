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
    parser.add_argument('--no-update', action='store_true',
                       help='Use local logs if they exist, skip downloading from remote')
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


def download_logs(log_dir: Path, skip_if_exists: bool = False):
    """Download logs from remote location.
    
    Args:
        log_dir: Path to the local log directory
        skip_if_exists: If True, skip download if log_dir already exists
    """
    if skip_if_exists and log_dir.exists():
        logger.info(f"Using existing local logs at {log_dir}")
        return
    
    remote_logs_path = os.getenv('REMOTE_LOGS')
    if not remote_logs_path:
        raise ValueError("REMOTE_LOGS environment variable is not set")
    
    logger.info(f"Downloading logs from {remote_logs_path}")
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
    positions_test, cfg2process, last_test_profit, last_real_profit,  = [], None, 0, 0    
    val_res = BackTestResults()
    val_res.plot_validation(y_label="Fin result")
    active_position = None
    for cfg_file in cfg_files + [None]:
        if cfg_file is None:
            cfg = {"date_start": np.datetime64("now")}
        else:
            # Start of the new cfg
            cfg = pickle.load(open(cfg_file, "rb"))

        if cfg2process is not None:
            backtest_trading = process_logfile(cfg2process, active_position)
            positions_test.extend(backtest_trading.session.positions)
            val_res.add(backtest_trading.session.profit_hist)
            val_res.add_profit_curve(backtest_trading.session.profit_hist.df.index,
                                     backtest_trading.session.profit_hist.df["profit_csum_nofees"]+last_test_profit,
                                     f"{cfg2process['symbol'].ticker} TEST" if cfg_file is None else None, "r", 1, 0.5)
            last_test_profit += backtest_trading.session.profit_hist.df["realized_pnl"].iloc[-1]

            profit_hist_real = TradeHistory(backtest_trading.session.mw, positions_real, active_position).df
            assert profit_hist_real.shape[0] > 0, "No real deals in history found"
            val_res.add_profit_curve(profit_hist_real["dates"], 
                                     profit_hist_real["profit_csum_nofees"]+last_real_profit,
                                     f"{cfg2process['symbol'].ticker} REAL" if cfg_file is None else None, "g", 3, 0.5)
            last_real_profit += profit_hist_real["realized_pnl"].iloc[-1]
            active_position = backtest_trading.session.active_position

        
        
        # if date_start is None:
        #     date_start = cfg_new["date_start"]
        # if cfg_file is None:
        #     cfg_optim = cfg2process.copy()
        #     cfg_optim["date_start"] = date_start
        #     positions_test.extend(process_logfile(cfg_optim, positions_real))    


        cfg2process = cfg
    val_res.save_fig()
    val_res.show_fig()
    return positions_test, positions_real


def process_logfile(cfg: Dict[str, Any], active_position: Position) -> tuple[list[Position]]:
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
    backtest_trading.session.set_start_active_position(active_position)
    backtest_trading.session.trade_stream(backtest_trading.handle_trade_message)

    return backtest_trading


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

    download_logs(log_dir, skip_if_exists=args.no_update)
    positions_test, positions_real = process_log_dir(log_dir)

    if len(positions_test) == 0:
        logger.warning(f"No positions while testing")

    positions_real.sort(key=lambda x: x.open_date)
    positions_test.sort(key=lambda x: x.open_date)

    time_lags, open_slippages, close_slippages = [], [], []
    open_slippages_rel, close_slippages_rel = [], []
    tp, fn, fp, prof_diff_summ = 0, 0, 0, 0

    logger.info("")
    logger.info(f"Validation for {len(positions_test)} positions")
    logger.info("----------------------------------------")
    logger.info(f"{'Open Date':<16} {'Side':<11} {'Open Slip,$':>15} {'Open Slip,%':>15} {'Time Lag,s':>15} {'Close Slip,$':>15} {'Close Slip,%':>15} {'Prof.diff,$':>15}")
    ir, sum_prof_real = 0, 0
    for pos_test in positions_test:
        date_test = pos_test.open_date.astype("datetime64[m]")
        logline, logline_suffix = f"{date_test} {pos_test.side:<4}", ""
        found_real = False
        for pos_real in positions_real[ir:]:
            date_real = pos_real.open_date.astype("datetime64[m]")
            if date_real == date_test and pos_real.side == pos_test.side:
                ir += 1
                tp += 1
                
                # Calculate open slippage
                open_diff = (pos_test.open_price - pos_real.open_price) * pos_test.side.value
                open_slippages.append(open_diff * pos_test.volume * pos_test.volume)
                open_slippages_rel.append(open_diff / pos_real.open_price * 100)
                logline += f" <- OK: {open_diff:15.2f} {open_slippages_rel[-1]:15.4f}"

                # Calculate open time lag
                time_lags.append((pos_real.open_date.astype("datetime64[ms]") -
                                  pos_test.open_date.astype("datetime64[ms]")).astype(int)/1000)
                logline += f" {time_lags[-1]:15.2f}"

                # Calculate close slippage
                close_diff = (pos_real.close_price - pos_test.close_price) * pos_test.side.value
                close_slippages.append(close_diff * pos_test.volume * pos_test.volume)
                close_slippages_rel.append(close_diff / pos_real.close_price * 100)
                logline += f" {close_diff:15.2f} {close_slippages_rel[-1]:15.4f}"

                prof_diff = pos_real.profit_abs - pos_test.profit_abs
                prof_diff_summ += prof_diff
                logline += f" {prof_diff:15.2f}"
                found_real = True
                sum_prof_real += pos_real.profit_abs
                break
            else:
                if date_real < date_test:
                    logger.info(f"{date_real.astype('datetime64[m]')} {pos_real.side:<4} -> NO TEST")
                    ir += 1
                    fn += 1
                else:
                    break
        if not found_real:
            logline += " <- NO REAL"
            fp += 1
        logger.info(logline + logline_suffix)

    mean_open_slippage = np.mean(open_slippages_rel) if len(open_slippages_rel) > 0 else 0
    mean_close_slippage = np.mean(close_slippages_rel) if len(close_slippages_rel) > 0 else 0

    logger.info(f"{'SUM VALUES: ':>29}" + f"{np.sum(open_slippages):15.2f} {mean_open_slippage:15.4f} {np.mean(time_lags):15.2f} {np.sum(close_slippages):15.2f} {mean_close_slippage:15.4f} {prof_diff_summ:15.2f}")
    logger.info("----------------------------------------")
    logger.info(f"Mean time lag:       " + (f"{np.mean(time_lags):.2f}s" if len(positions_test) > 0 else "NO TEST MATCHES"))
    logger.info(f"Match rate (accuracy):          " + (f"{tp / (tp + fn + fp) * 100:.2f}%" if len(positions_test) > 0 else "NO TEST MATCHES"))
    logger.info(f"Mean relative open slippage:  " + (f"{mean_open_slippage:.4f}%" if len(open_slippages_rel) > 0 else "NO TEST MATCHES"))
    logger.info(f"Mean relative close slippage: " + (f"{mean_close_slippage:.4f}%" if len(close_slippages_rel) > 0 else "NO TEST MATCHES"))
    logger.info(f"Relative profit diff:   " + (f"{prof_diff_summ / sum_prof_real * 100:.4f}%" if sum_prof_real > 0 else "NO TEST MATCHES"))
    logger.info("\n* negative values mean the real position was opened/closed at a worse price than the test position")
    logger.info("* negative profit diff means the real position has less profit than the test position\n\n")
