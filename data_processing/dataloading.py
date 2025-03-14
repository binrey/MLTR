import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm

from common.type import to_datetime

# Define the dtype for the structured array
DTYPE = [('Date', np.dtype('<M8[m]')), 
         ('Open', np.dtype('float64')), 
         ('High', np.dtype('float64')), 
         ('Low', np.dtype('float64')), 
         ('Close', np.dtype('float64')), 
         ('Volume', np.dtype('float64')),
         ('Id', np.dtype('int64'))]
        
def build_features(f, dir, sl, rate, open_date=None, timeframe=None):
    fo = f.Open/f.Open[-1]
    fc = f.Close/f.Open[-1]
    fh = f.High/f.Open[-1]
    fl = f.Low/f.Open[-1]
    fv = f.Volume[:-2] / \
        f.Volume[-2] if f.Volume[-2] != 0 else np.ones_like(f.Volume[:-2])

    if dir > 0:
        x = np.vstack([fc, fo, fl, fh])
    else:
        x = np.vstack([2-fc, 2-fo, 2-fl, 2-fh])
    x = x*127
    # x = np.vstack([x, fv])
    # x = np.vstack([x, np.ones(x.shape[1])*sl*10])
    # x = np.vstack([x, np.ones(x.shape[1])*rate*1000])
    if open_date is not None:
        odate = to_datetime(open_date)
        odate = odate.year*10000 + odate.month*100 + odate.day
        x = np.vstack([x, np.ones(x.shape[1])*odate])
    if timeframe is not None:
        x = np.vstack([x, np.ones(x.shape[1])*timeframe])
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_data(X, y, test_split=0.25, n1_split=0, n2_split=1, period2process=0):
    ids = np.arange(X.shape[0])
    # np.random.shuffle(ids)

    ids_test, periods_test, odates_testset, odates, periods = [
    ], [], set(), X[:, 0, -2, 0], X[:, 0, -1, 0]
    odates_set = sorted(list(set(odates.astype(int))))
    dates_count = len(odates_set)
    test_size = int(dates_count*test_split)
    if test_size*n2_split > dates_count:
        return None
    di0, di1 = test_size*n1_split, test_size*(n2_split) - 1
    name = f"test dates: {odates_set[di0]}:{odates_set[di1]}"
    for di in tqdm(range(di0, di1), name):
        d = odates_set[di]
        if d not in odates_testset:
            selected_days = odates == d
            ii = ids[selected_days]
            periods_test += periods[selected_days].tolist()
            ids_test += ii.tolist()
            odates_testset.add(d)
        di += 1
    periods_test = np.array(periods_test)
    ids_test = np.array(ids_test)
    ids_train = [ix for ix in ids if ix not in ids_test]
    ids_test = ids_test[periods_test == period2process]
    periods_test = periods_test[periods_test == period2process]
    np.random.shuffle(ids_train)
    np.random.shuffle(ids_test)

    X_train, X_test, y_train, y_test, profs_train, profs_test = X[ids_train], X[
        ids_test], y[ids_train], y[ids_test], y[ids_train].copy(), y[ids_test].copy()
    X_train = X_train[:, :, :-2, :].astype(np.uint8)
    X_test = X_test[:, :, :-2, :].astype(np.uint8)

    # y_train = np.eye(3)[np.argmax(y_train, 1).reshape(-1)].astype(np.float32)
    # y_test = np.eye(3)[np.argmax(y_test, 1).reshape(-1)].astype(np.float32)

    return X_train, X_test, y_train, y_test, profs_train, profs_test, periods_test, (str(odates_set[di0]), str(odates_set[di1]))


class DataParser():
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, database=None):
        if database is None:
            database = os.environ.get("FINDATA", "../fin_data")
        t0 = perf_counter()
        p = Path(database) / self.cfg["data_type"] / self.cfg["period"].value
        flist = [f for f in p.glob("*") if self.cfg["symbol"].ticker in f.stem]
        if len(flist):
            fpath = flist[np.argmin([len(f.name) for f in flist])]
            data = {"metatrader": self.metatrader,
                    "FORTS": self.metatrader,
                    "bitfinex": self.bitfinex,
                    "yahoo": self.yahoo,
                    "bybit": self.bybit
                    }.get(self.cfg["data_type"], None)(fpath)
            logger.info(f"Loaded {self.cfg['data_type']} data from {fpath} in {perf_counter() - t0:.1f} sec")
            return data
        else:
            raise FileNotFoundError(f"No data for {self.cfg['symbol'].ticker} in {p}")
    
    def bybit(self, data_file):
        pd.options.mode.chained_assignment = None
        hist = pd.read_csv(data_file, sep=",")
        hist["Date"] = np.array(hist.Date.values).astype("datetime64[m]")
        hist["Id"] = list(range(hist.shape[0]))
        hist_structured = np.array([tuple(row) for row in hist.to_records(index=False)], dtype=DTYPE)
        return hist_structured

    def metatrader(self, data_file):
        pd.options.mode.chained_assignment = None
        hist = pd.read_csv(data_file, sep=",")
        hist.TIME = to_datetime(hist.TIME, utc=True)
        hist.columns = map(str.capitalize, hist.columns)
        # if "Time" not in hist.columns:
        #     hist["Time"] = ["00:00:00"]*hist.shape[0]
        # hist["Date"] = to_datetime([" ".join([d, t]) for d, t in zip(
        #     hist.Date.values, hist.Time.values)], utc=True)
        hist.drop(["Tick_volume", "Spread"], axis=1, inplace=True)
        hist.rename(columns={"Real_volume": "Volume", "Time": "Date"}, inplace=True)
        hist["Id"] = list(range(hist.shape[0]))
        hist_dict = EasyDict({c: hist[c].values for c in hist.columns})
        # hist_dict["Date"] = hist["Date"].values
        hist.set_index("Date", inplace=True, drop=True)
        return hist, hist_dict

    def bitfinex(self, data_file):
        hist = pd.read_csv(data_file, header=1)
        hist = hist[::-1]
        hist["Date"] = to_datetime(hist.date.values)
        hist.set_index("Date", inplace=True, drop=False)
        hist["Id"] = list(range(hist.shape[0]))
        hist.drop(["unix", "symbol", "date"], axis=1, inplace=True)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Volume"] = hist.iloc[:, -3]
        hist_dict = EasyDict({c: hist[c].values for c in hist.columns})
        return hist, hist_dict

    def yahoo(self, data_file):
        hist = pd.read_csv(data_file)
        hist["Date"] = to_datetime(hist.Date, utc=True)
        hist["Id"] = list(range(hist.shape[0]))
        hist_dict = EasyDict({c: hist[c].values for c in hist.columns})
        hist.set_index("Date", inplace=True, drop=True)
        return hist, hist_dict


def collect_train_data(dir, fsize=64, glob="*.pickle"):
    cfgs, btests = [], []
    for p in sorted(Path(dir).glob(glob)):
        cfg, btest = pickle.load(open(p, "rb"))
        cfgs.append(cfg)
        btests.append(btest)
    print(len(btests))

    tfdict = {"M5": 0, "M15": 1, "H1": 2}
    X, y = [], []
    for btest in tqdm(btests, "Load pickles"):
        # print(btest.cfg.ticker, end=" ")
        hist_pd, hist = DataParser(btest.cfg).load()
        mw = MovingWindow(hist, fsize)
        # print(len(btest.positions))
        for pos in btest.positions:
            f, _ = mw(pos.open_indx)
            x = build_features(f,
                               pos.dir,
                               btest.cfg.stops_processor.func.cfg.sl,
                               btest.cfg.rate,
                               pos.open_date,
                               tfdict[btest.cfg.period.value])
            X.append([x])
            y.append(pos.profit)

    X, y = np.array(X), np.array(y, dtype=np.float32)
    print(X.shape, y.shape)
    print(f"{X[0, 0, -2, 0]:8.0f} -> {X[-1, 0, -2, 0]:8.0f}")
    return X, y


def collect_train_data2(dir, fsize=64, nparams=4):
    cfgs, btests = [], []
    for p in sorted(Path(dir).glob("*.pickle")):
        cfg, btest = pickle.load(open(p, "rb"))
        cfgs.append(cfg)
        btests.append(btest)
    print(len(btests))

    tfdict = {"M5": 0, "M15": 1, "H1": 2}
    X, y = [], []
    posdict = {}
    for btest in tqdm(btests, "Load pickles"):
        for pos in btest.positions[4:]:
            k = pos.open_date
            sl, profit, dir, pos_id = btest.cfg.stops_processor.func.cfg.sl, pos.profit, pos.dir, pos.open_indx
            if k in posdict.keys():
                posdict[k]["sl"].append(sl)
                posdict[k]["prof"].append(profit)
                posdict[k]["dir"].append(dir)
                posdict[k]["id"].append(pos_id)
            else:
                posdict[k] = {"sl": [sl], "prof": [
                    profit], "dir": [dir], "id": [pos_id]}

    btest = btests[0]
    hist_pd, hist = DataParser(btest.cfg).load()
    mw = MovingWindow(hist, fsize+2)
    for open_date, pos in posdict.items():
        if len(set(pos["sl"])) == nparams and len(set(pos["dir"])) == 1:
            f, _ = mw(pos["id"][0])
            x = build_features(f,
                               pos["dir"][0],
                               0,
                               btest.cfg.rate,
                               open_date,
                               tfdict[btest.cfg.period.value])
            X.append([x])
            y.append(pos["prof"])

    X, y = np.array(X), np.array(y, dtype=np.float32)
    print(X.shape, y.shape)
    print(f"{X[0, 0, -2, 0]:8.0f} -> {X[-1, 0, -2, 0]:8.0f}")
    return X, y


class MovingWindow():
    def __init__(self, cfg):
        self.hist = DataParser(cfg).load()
        self.date_start = np.datetime64(cfg["date_start"])
        self.date_end = np.datetime64(cfg["date_end"])
        self.size = cfg["hist_buffer_size"]
        self.ticker = cfg["symbol"].ticker

        self.id2start = self.find_nearest_date_indx(
            self.hist["Date"], self.date_start)
        if self.id2start == self.hist["Id"][-1]:
            logger.error(f"Date start {self.date_start} is equal or higher than latest range date {self.hist['Date'][-1]}")
            raise ValueError()

        if self.id2start < self.size:
            logger.warning(f"Not enough history, shift start id from {self.id2start} to {self.size}")
            self.id2start = self.size
            logger.warning(
            f"Switch to {self.hist['Date'][self.id2start]}")

        self.id2end = self.find_nearest_date_indx(
            self.hist["Date"], self.date_end)
        
        self.date_start = self.hist["Date"][self.id2start]
        self.date_end = self.hist["Date"][self.id2end]

    @staticmethod
    def find_nearest_date_indx(array, target):
        idx = (np.abs(array - target)).argmin()
        return idx

    @property
    def timesteps_count(self):
        return self.id2end - self.id2start

    def __getitem__(self, t):
        t0 = perf_counter()
        data = self.hist[t-self.size+1:t+1].copy()
        data['Close'][-1] = data['Open'][-1]
        data['High'][-1] = data['Open'][-1]
        data['Low'][-1] = data['Open'][-1]
        data["Volume"][-1] = 0
        return data, perf_counter() - t0

    def __call__(self, output_time=True):
        logger.info(f"Start generate {self.ticker} data from {self.date_start} (id:{self.id2start}) to {self.date_end} (id:{self.id2end})")
        # for t in tqdm(range(self.id2start, self.id2end), desc=f"Processing {self.ticker}"):
        for t in range(self.id2start, self.id2end):    
            yield self[t] if output_time else self[t][0]


if __name__ == "__main__":
    from dataloading import collect_train_data
    X, y = collect_train_data("./optimization/", 64)
