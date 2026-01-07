from datetime import datetime
from pathlib import Path
from typing import Optional

import mplfinance as mpf
import pandas as pd
from loguru import logger
from pybit.unified_trading import HTTP

from common.type import Symbol, TimePeriod
from data_processing.dataloading import get_bybit_hist


class BybitDownloader:
    def __init__(self, symbol: Symbol, period: TimePeriod, init_data=None):
        self.symbol = symbol.ticker
        self.period = period
        self.init_data = init_data
        self.init_dirs()

    def init_dirs(self):
        if self.init_data is not None:
            self.init_data = Path(self.init_data)
            if not self.init_data.exists():
                self.init_data.parent.mkdir(parents=True, exist_ok=True)
                self.init_data.touch()
                self.init_data.write_text("Date,Open,High,Low,Close,Volume\n")

    def get_klines(self, start_date, size: int=1000, end_date: Optional[datetime]=None):
        if start_date is not None and isinstance(start_date, datetime):
            start_date=start_date.timestamp()*1000
        if end_date is not None and isinstance(end_date, datetime):
            end_date=end_date.timestamp()*1000

        session = HTTP()
        message = session.get_kline(
            category="linear",
            symbol=self.symbol,
            interval=str(self.period.minutes),
            start=start_date,
            end=end_date,
            limit=size
        )
        return message["result"]

    @staticmethod
    def _get_data_from_message(mresult):
        data = get_bybit_hist(mresult)
        data = pd.DataFrame(data)
        data.set_index(data.Date, drop=True, inplace=True)
        data.drop("Date", axis=1, inplace=True)
        return data


    def read_from_file(self, filename):
        data = pd.read_csv(filename)
        data.Date = pd.to_datetime(data.Date, format="mixed")
        data.set_index(data.Date, drop=True, inplace=True)
        data.drop("Date", axis=1, inplace=True)
        return data

    def get_history(self, date_start: Optional[pd.Timestamp] = None, date_end: Optional[pd.Timestamp] = None):
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end) if date_end is not None else pd.Timestamp.now()
        
        logger.info(
            f"Pulling data for {self.symbol} from {date_start} to {date_end}")
        if self.init_data is not None:
            h = self.read_from_file(self.init_data)
        else:
            h = self._get_data_from_message(self.get_klines(start_date=date_start,
                                                            end_date=date_end))
        start_saved_date = h.index[0] if len(h) > 0 else None
        end_saved_date = h.index[-1] if len(h) > 0 else None
        logger.info(f"Start date: {start_saved_date}")
        logger.info(f"Last date : {end_saved_date}")

        if start_saved_date is not None and end_saved_date is not None:
            if date_start >= start_saved_date and date_end <= end_saved_date:
                logger.info(f"Data already exists in database")
                return h

        res = self._get_data_from_message(
            self.get_klines(start_date=h.index[-1] if len(h) > 0 else date_start,
                            end_date=None))
        if len(h) > 0:
            res = res[1:]
        h = pd.concat([h, res])
        while res.shape[0] and h.index[-1] < date_end:
            logger.info(f"Last date : {h.index[-1]}")
            res = self._get_data_from_message(self.get_klines(start_date=h.index[-1],
                                                              end_date=None))[1:]
            h = pd.concat([h, res])
        h.to_csv(self.init_data)
        return h


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--start_date", type=str, default="2025-04-20")
    parser.add_argument("--init_data", type=str, default=None)
    args = parser.parse_args()

    bb_loader = BybitDownloader(symbol=args.symbol,
                                period=args.period,
                                start_date=args.start_date,
                                init_data=args.init_data
                                )
    hist = bb_loader.get_history().to_csv(args.init_data)

    print(hist)
    mpf.plot(hist, type='line')
