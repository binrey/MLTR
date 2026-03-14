import json
import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import PyConfig
from loguru import logger
from trade.backtest import launch as backtest_launch


FIXTURE_PATH = REPO_ROOT / "tests" / "integration" / "fixtures" / "macross_ground_truth.json"
COMMITTED_DATE_START = "2025-01-01T00:00:00"
COMMITTED_DATE_END = "2026-01-01T00:00:00"
COMMITTED_FINDATA = str(REPO_ROOT / "fin_data")
CONFIG_PATHS = {
    "BTCUSDT": REPO_ROOT / "configs" / "macross" / "BTCUSDT.py",
    "ETHUSDT": REPO_ROOT / "configs" / "macross" / "ETHUSDT.py",
}

FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-6


def ensure_runtime_env() -> None:
    os.environ["FINDATA"] = COMMITTED_FINDATA
    os.environ.setdefault("LOG_DIR", str(REPO_ROOT / "logs"))
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("TEST_LOG_LEVEL", "ERROR"))


def collect_core_metrics(config_path: Path) -> dict:
    ensure_runtime_env()
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


class TestMACrossGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FIXTURE_PATH.exists():
            raise FileNotFoundError(f"Missing fixture file: {FIXTURE_PATH}")
        cls.expected = json.loads(FIXTURE_PATH.read_text())
        params = cls.expected.get("_params", {})
        if not params:
            raise AssertionError("Fixture is missing required '_params' section")
        if (
            params.get("date_start") != COMMITTED_DATE_START
            or params.get("date_end") != COMMITTED_DATE_END
            or params.get("findata") != COMMITTED_FINDATA
        ):
            raise AssertionError(
                "Committed parameters mismatch between test and fixture: "
                f"expected dates {COMMITTED_DATE_START}..{COMMITTED_DATE_END}, findata {COMMITTED_FINDATA}; got "
                f"dates {params.get('date_start')}..{params.get('date_end')}, findata {params.get('findata')}"
            )

    def test_core_metrics_match_snapshot(self):
        for symbol, config_path in CONFIG_PATHS.items():
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, self.expected, f"Missing '{symbol}' section in fixture")
                expected_metrics = self.expected[symbol]
                actual_metrics = collect_core_metrics(config_path)

                self.assertEqual(
                    actual_metrics["ndeals"],
                    int(expected_metrics["ndeals"]),
                    f"{symbol}: ndeals mismatch",
                )

                for metric_name in ("APR", "final_profit", "max_drawdown", "recovery_factor"):
                    self.assertTrue(
                        math.isclose(
                            actual_metrics[metric_name],
                            float(expected_metrics[metric_name]),
                            rel_tol=FLOAT_REL_TOL,
                            abs_tol=FLOAT_ABS_TOL,
                        ),
                        (
                            f"{symbol}: {metric_name} mismatch. "
                            f"expected={expected_metrics[metric_name]}, actual={actual_metrics[metric_name]}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
