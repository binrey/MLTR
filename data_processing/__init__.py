import os
from pathlib import Path
from typing import Any

from common.type import Symbol, TimePeriod


def pull_bybit_data(symbol: Symbol, period: TimePeriod, date_start: Any, date_end: Any, **kwargs):
    from data_processing.datapulling import BybitDownloader

    init_data = Path(os.getenv("FINDATA")) / f"bybit/{period.value}/{symbol.ticker}.csv"
    bb_loader = BybitDownloader(symbol=symbol,
                                period=period,
                                init_data=init_data
                                )
    h = bb_loader.get_history(date_start=date_start, date_end=date_end)
    return h


def pull_yahoo_data(symbol: Symbol, date_start: Any, date_end: Any, yahoo_ticker: str | None = None, **kwargs):
    from data_processing.datapulling import YahooDownloader

    out = Path(os.getenv("FINDATA")) / f"yahoo/D/{symbol.ticker}.csv"
    dl = YahooDownloader(
        symbol=symbol,
        period=TimePeriod.D,
        init_data=out,
        yahoo_ticker=yahoo_ticker,
    )
    return dl.get_history(date_start=date_start, date_end=date_end)


PULLERS = {
    "bybit": pull_bybit_data,
    "yahoo": pull_yahoo_data,
}