import os
from pathlib import Path
from typing import Any

from common.type import Symbol, TimePeriod
from data_processing.datapulling import BybitDownloader


def pull_bybit_data(symbol: Symbol, period: TimePeriod, date_start: Any, date_end: Any, **kwargs):
    init_data = Path(os.getenv("FINDATA")) / f"bybit/{period.value}/{symbol.ticker}.csv"
    bb_loader = BybitDownloader(symbol=symbol,
                                period=period,
                                init_data=init_data
                                )
    h = bb_loader.get_history(date_start=date_start, date_end=date_end)
    return h


PULLERS = {
    "bybit": pull_bybit_data
}