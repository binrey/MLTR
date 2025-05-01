from datetime import datetime
from pathlib import Path

import mplfinance as mpf
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

from common.type import Symbol, TimePeriod


class BybitDownloader:
    def __init__(self, symbol: Symbol, period: TimePeriod, start_date=None, init_data=None):
        self.symbol = symbol.ticker
        self.period = period.minutes
        self.init_data = init_data
        self.start_date = start_date if isinstance(start_date, pd.Timestamp) else pd.to_datetime(start_date)
        self.init_dirs()

    def init_dirs(self):
        if self.init_data is not None:
            self.init_data = Path(self.init_data)
            if not self.init_data.exists():
                self.init_data.parent.mkdir(parents=True, exist_ok=True)
                self.init_data = None

    def get_klines(self, start_date, size: int=1000, end: datetime=None):
        if start_date is not None and isinstance(start_date, datetime):
            start_date=start_date.timestamp()*1000
        if end is not None and isinstance(end, datetime):
            end=end.timestamp()*1000

        session = HTTP()
        message = session.get_kline(
            category="linear",
            symbol=self.symbol,
            interval=str(self.period),
            start=start_date,
            end=end,
            limit=size
        )
        return message["result"]

    @staticmethod
    def _get_data_from_message(mresult):
        data = {}
        input = np.array(mresult["list"], dtype=np.float64)[::-1]
        data["Date"] = pd.to_datetime(input[:, 0].astype(np.int64)*1000000)
        data["Open"] = input[:, 1]
        data["High"] = input[:, 2]
        data["Low"]  = input[:, 3]
        data["Close"]= input[:, 4]
        data["Volume"] = input[:, 5]
        data = pd.DataFrame(data)
        data.set_index(data.Date, drop=True, inplace=True)
        data.drop("Date", axis=1, inplace=True)
        return data

    def read_from_file(self, filename):
        # filename - csv file name
        data = pd.read_csv(filename)
        data.Date = pd.to_datetime(data.Date, format="mixed")
        data.set_index(data.Date, drop=True, inplace=True)
        data.drop("Date", axis=1, inplace=True)
        return data

    def get_history(self):
        if self.init_data is not None:
            h = self.read_from_file(self.init_data)
            self.start_date = h.index[-1]
        else:
            h = self._get_data_from_message(self.get_klines(start_date=self.start_date))
        print(h.index[-1])
        res = self._get_data_from_message(self.get_klines(start_date=h.index[-1]))[1:]
        h = pd.concat([h, res])
        while res.shape[0]:
            print(h.index[-1])
            res = self._get_data_from_message(self.get_klines(start_date=h.index[-1]))[1:]
            h = pd.concat([h, res])
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
