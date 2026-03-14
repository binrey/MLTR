#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import PyConfig
from loguru import logger
from trade.backtest import launch as backtest_launch


DEFAULT_OUTPUT = REPO_ROOT / "tests" / "integration" / "fixtures" / "macross_ground_truth.json"
COMMITTED_DATE_START = "2025-01-01T00:00:00"
COMMITTED_DATE_END = "2026-01-01T00:00:00"
COMMITTED_FINDATA = str(REPO_ROOT / "fin_data")
CONFIG_PATHS = {
    "BTCUSDT": REPO_ROOT / "configs" / "macross" / "BTCUSDT.py",
    "ETHUSDT": REPO_ROOT / "configs" / "macross" / "ETHUSDT.py",
}


def ensure_runtime_env() -> None:
    os.environ["FINDATA"] = COMMITTED_FINDATA
    os.environ.setdefault("LOG_DIR", str(REPO_ROOT / "logs"))
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("TEST_LOG_LEVEL", "ERROR"))


def collect_core_metrics(config_path: Path) -> dict:
    cfg = PyConfig(str(config_path)).get_backtest()
    cfg["date_start"] = COMMITTED_DATE_START
    cfg["date_end"] = COMMITTED_DATE_END
    bt_res = backtest_launch(cfg)
    return {
        "APR": float(bt_res.APR),
        "final_profit": float(bt_res.final_profit),
        "ndeals": int(bt_res.ndeals),
        "max_drawdown": float(bt_res.metrics.max_drawdown),
        "recovery_factor": float(bt_res.metrics.recovery_factor),
    }


def main() -> int:
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    ensure_runtime_env()

    snapshot = {
        "_params": {
            "date_start": COMMITTED_DATE_START,
            "date_end": COMMITTED_DATE_END,
            "findata": COMMITTED_FINDATA,
        }
    }
    for symbol, config_path in CONFIG_PATHS.items():
        snapshot[symbol] = collect_core_metrics(config_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"Ground-truth snapshot saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
