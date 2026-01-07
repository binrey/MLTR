import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml
from pybit.unified_trading import HTTP

from common.utils import PyConfig, Telebot
from trade.bybit import BybitTrading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test connection to broker via BaseTradeClass.update_market_state()"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to config file (e.g. configs/volprof/ETHUSDT.py)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo credentials from api.yaml (bybit_demo section)",
    )
    return parser.parse_args()


def build_trader(cfg_path: Path, demo: bool) -> BybitTrading:
    cfg = PyConfig(str(cfg_path)).get_bybit()

    with open("./api.yaml", "r") as f:
        api = yaml.safe_load(f)

    bybit_creds = api["bybit_demo"] if demo else api[cfg["credentials"]]
    bot_token = api["bot_token"]

    bybit_session = HTTP(
        testnet=False,
        api_key=bybit_creds["api_key"],
        api_secret=bybit_creds["api_secret"],
        demo=demo,
    )

    telebot = Telebot(bot_token)
    trader = BybitTrading(cfg=cfg, telebot=telebot, bybit_session=bybit_session)
    return trader


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    trader = build_trader(cfg_path, args.demo)

    logger.info("Calling update_market_state() to test connection...")
    trader.update_market_state()

    logger.info(f"Deposit from broker: {trader.deposit}")
    logger.info(f"Current position: {trader.pos.curr}")

    print("OK: update_market_state() completed")
    print(f"Deposit: {trader.deposit}")
    print(f"Current position: {trader.pos.curr}")


if __name__ == "__main__":
    main()


