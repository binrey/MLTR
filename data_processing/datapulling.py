import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import mplfinance as mpf
import pandas as pd
from loguru import logger
from pybit.unified_trading import HTTP

from common.type import Symbol, TimePeriod
from data_processing.dataloading import get_bybit_hist


load_dotenv(override=True)


# Yahoo Finance uses hyphenated tickers (e.g. BTC-USD); there is no BTC/USDT symbol on Yahoo.
YAHOO_CRYPTO_BYBIT = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "XRPUSDT": "XRP-USD",
    "SOLUSDT": "SOL-USD",
}


def bybit_symbol_to_yahoo_ticker(bybit_ticker: str) -> str:
    if bybit_ticker in YAHOO_CRYPTO_BYBIT:
        return YAHOO_CRYPTO_BYBIT[bybit_ticker]
    if "-" in bybit_ticker:
        return bybit_ticker
    raise ValueError(
        f"No Yahoo mapping for {bybit_ticker!r}. Use a key from {sorted(YAHOO_CRYPTO_BYBIT)} "
        "or pass an explicit Yahoo ticker (e.g. BTC-USD)."
    )


class YahooDownloader:
    """Daily OHLCV from Yahoo Finance; CSV layout matches ``DataParser.yahoo``."""

    def __init__(
        self,
        symbol: Symbol,
        period: TimePeriod,
        init_data=None,
        yahoo_ticker: Optional[str] = None,
    ):
        self.symbol = symbol.ticker
        self.period = period
        self.yahoo_ticker = yahoo_ticker or bybit_symbol_to_yahoo_ticker(symbol.ticker)
        self.init_data = init_data
        self.init_dirs()

    def init_dirs(self):
        if self.init_data is not None:
            self.init_data = Path(self.init_data)
            if not self.init_data.exists():
                self.init_data.parent.mkdir(parents=True, exist_ok=True)
                self.init_data.touch()
                self.init_data.write_text("Date,Open,High,Low,Close,Volume\n")

    def read_from_file(self, filename):
        data = pd.read_csv(filename)
        data["Date"] = pd.to_datetime(data["Date"], format="mixed")
        data.set_index("Date", drop=True, inplace=True)
        return data

    @staticmethod
    def _yfinance():
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("Install yfinance in your environment: pip install yfinance") from e
        return yf

    def _fetch(self, date_start: pd.Timestamp, date_end: pd.Timestamp) -> pd.DataFrame:
        """Return OHLCV indexed by naive daily ``Date`` (inclusive range)."""
        yf = self._yfinance()
        t = yf.Ticker(self.yahoo_ticker)
        kwargs: dict = {"interval": "1d", "auto_adjust": False, "start": date_start.strftime("%Y-%m-%d")}
        # Yahoo ``end`` is exclusive; add one calendar day so ``date_end`` is included.
        kwargs["end"] = (date_end.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = t.history(**kwargs)
        if df.empty:
            logger.warning(
                f"No rows returned for {self.yahoo_ticker!r} "
                f"(start={date_start!r}, end={date_end!r})"
            )
            return df
        df = df.rename_axis("Date").reset_index()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
        ohlcv = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[["Date", *ohlcv]]
        df.set_index("Date", drop=True, inplace=True)
        return df

    def _write_csv(self, hist: pd.DataFrame) -> None:
        out = hist.reset_index()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(self.init_data, index=False)
        logger.info(f"Wrote {len(out)} rows to {self.init_data}")

    def get_history(
        self,
        date_start: Optional[pd.Timestamp] = None,
        date_end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if self.init_data is None:
            raise ValueError("init_data path is required for YahooDownloader")
        if self.period != TimePeriod.D:
            raise ValueError(f"YahooDownloader only supports TimePeriod.D, got {self.period!r}")

        date_start = pd.to_datetime(date_start).normalize()
        date_end = pd.to_datetime(date_end).normalize() if date_end is not None else pd.Timestamp.now().normalize()

        logger.info(
            f"Pulling Yahoo {self.yahoo_ticker} ({self.symbol}) from {date_start} to {date_end}"
        )

        h = pd.DataFrame()
        if self.init_data.exists() and self.init_data.stat().st_size > 0:
            try:
                h = self.read_from_file(self.init_data)
                if len(h.index):
                    h.index = pd.to_datetime(h.index).normalize()
            except (pd.errors.EmptyDataError, ValueError):
                h = pd.DataFrame()

        if len(h) > 0:
            start_saved = h.index[0]
            end_saved = h.index[-1]
            logger.info(f"Start date: {start_saved}, Last date: {end_saved}")
            if date_start >= start_saved and date_end <= end_saved:
                logger.info("Data already exists in database")
                return h.loc[date_start:date_end]

        pull_start = min(date_start, h.index[0]) if len(h) > 0 else date_start
        pull_end = max(date_end, h.index[-1]) if len(h) > 0 else date_end
        new_df = self._fetch(pull_start, pull_end)
        if new_df.empty and len(h) == 0:
            return new_df
        if not new_df.empty:
            h = pd.concat([h, new_df]).sort_index()
            h = h[~h.index.duplicated(keep="last")]
            self._write_csv(h)
        return h.loc[date_start:date_end]


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
            interval=self.period.bybit_interval,
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

    @staticmethod
    def _drop_bars_not_after(res: pd.DataFrame, last_ts: pd.Timestamp) -> pd.DataFrame:
        """Remove overlap with data we already have; Bybit returns bars after ``start``, not a repeat of it."""
        if res.empty:
            return res
        return res[res.index > last_ts]


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
        if len(h) > 0:
            start_saved_date = h.index[0]
            end_saved_date = h.index[-1]
            logger.info(f"Found {len(h)} rows in database, start date: {start_saved_date}, end date: {end_saved_date}")
            if date_start >= start_saved_date and date_end <= end_saved_date:
                logger.info(f"All Data already exists in database")
                return h.loc[date_start:date_end]

        start_from = h.index[-1] if len(h) > 0 else date_start
        chunks = []

        res = self._get_data_from_message(
            self.get_klines(start_date=start_from, end_date=None)
        )
        if len(h) > 0 and not res.empty:
            res = self._drop_bars_not_after(res, h.index[-1])
        if not res.empty:
            chunks.append(res)

        last_loaded = res.index[-1] if not res.empty else (h.index[-1] if len(h) > 0 else date_start)
        while not res.empty and last_loaded < date_end:
            logger.info(f"Last date : {last_loaded}")
            res = self._get_data_from_message(
                self.get_klines(start_date=last_loaded, end_date=None)
            )
            if not res.empty:
                res = self._drop_bars_not_after(res, last_loaded)
            if res.empty:
                break
            chunks.append(res)
            last_loaded = res.index[-1]

        if chunks:
            h = pd.concat([h] + chunks)
        h.to_csv(self.init_data)
        return h


if __name__ == "__main__":
    from argparse import ArgumentParser

    from common.type import Symbols
    from common.utils import resolve_findata_dir

    period_names = [p.name for p in TimePeriod]

    parser = ArgumentParser(
        description="Download market data to CSV. Output path is <FINDATA or fin_data>/<bybit|yahoo>/<period>/<symbol>.csv."
    )
    parser.add_argument(
        "--yahoo",
        action="store_true",
        help="Download from Yahoo Finance (daily bars). --symbol selects the pair (see YAHOO_CRYPTO_BYBIT mapping).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbols attribute name (e.g. BTCUSDT, ETHUSDT); used for Yahoo ticker mapping and output filename.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="D",
        choices=period_names,
        help=f"TimePeriod name: {', '.join(period_names)}",
    )
    parser.add_argument("--start_date", type=str, default="2017-01-01")
    parser.add_argument("--end_date", type=str, default=None, help="Default: now")
    args = parser.parse_args()
    data_type = "yahoo" if args.yahoo else "bybit"

    if not hasattr(Symbols, args.symbol):
        raise SystemExit(f"Unknown symbol {args.symbol!r}; use a name from Symbols (e.g. BTCUSDT).")
    symbol = getattr(Symbols, args.symbol)
    period = TimePeriod[args.period]  
    init_data = resolve_findata_dir() / data_type / period.value / f"{symbol.ticker}.csv"

  
    if args.yahoo:

        if period != TimePeriod.D:
            raise SystemExit("Yahoo daily download only supports period D for now.")

        yh = YahooDownloader(symbol=symbol, period=period, init_data=init_data)
        hist = yh.get_history(date_start=args.start_date, date_end=args.end_date)
        if hist.empty:
            raise SystemExit(1)
        hist.index.name = "Date"
        print(hist.tail())
        mpf.plot(hist, type="line")
    else:
        period = TimePeriod[args.period]

        bb_loader = BybitDownloader(symbol=symbol, period=period, init_data=init_data)
        hist = bb_loader.get_history(date_start=args.start_date, date_end=args.end_date)
        print(hist.tail())
        mpf.plot(hist, type="line")
