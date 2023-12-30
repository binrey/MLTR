import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from easydict import EasyDict
from pathlib import Path
from time import perf_counter
from loguru import logger
from tqdm import tqdm
import pickle


def build_features(f, dir, sl, trailing_stop_rate, open_date=None, timeframe=None):
    fo = f.Open/f.Open[-1]
    fc = f.Close/f.Open[-1]
    fh = f.High/f.Open[-1]
    fl = f.Low/f.Open[-1]
    fv = f.Volume[:-2]/f.Volume[-2] if f.Volume[-2] != 0 else np.ones_like(f.Volume[:-2])

    if dir > 0:
        x = np.vstack([fc, fo, fl, fh])
    else:
        x = np.vstack([2-fc, 2-fo, 2-fl, 2-fh])
    x = x*127
    # x = np.vstack([x, fv])
    # x = np.vstack([x, np.ones(x.shape[1])*sl])
    # x = np.vstack([x, np.ones(x.shape[1])*trailing_stop_rate*1000])
    if open_date is not None:
        odate = pd.to_datetime(open_date)
        odate = odate.year*10000 + odate.month*100 + odate.day       
        x = np.vstack([x, np.ones(x.shape[1])*odate])
    if timeframe is not None:
        x = np.vstack([x, np.ones(x.shape[1])*timeframe])
    return x.astype(np.uint8)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def get_data(X, y, test_period=0, test_split=0.25, n1_split=0, n2_split=1):
    ids = np.arange(X.shape[0])
    # np.random.shuffle(ids)
    
    ids_test, periods_test, odates_testset, odates, periods = [], [], set(), X[:, 0, -2, 0], X[:, 0, -1, 0]
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
    ids_test = ids_test[periods_test == test_period]
    periods_test = periods_test[periods_test == test_period]
    np.random.shuffle(ids_train) 
    np.random.shuffle(ids_test) 
        
    X_train, X_test, y_train, y_test, profs_train, profs_test = X[ids_train], X[ids_test], y[ids_train], y[ids_test], y[ids_train].copy(), y[ids_test].copy()
    X_train = X_train[:, :, :-1, :]
    X_test = X_test[:, :, :-1, :]
    
    # y_train = np.eye(3)[np.argmax(y_train, 1).reshape(-1)].astype(np.float32)
    # y_test = np.eye(3)[np.argmax(y_test, 1).reshape(-1)].astype(np.float32)
    
    return X_train, X_test, y_train, y_test, profs_train, profs_test, periods_test, (str(odates_set[di0]), str(odates_set[di1]))

class CustomImageDataset(Dataset):
    def __init__(self, X, y):
        self.img_labels = y
        self.imgs = X

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label
    

class DataParser():
    def __init__(self, cfg):
        self.cfg = cfg     
        
    def load(self):
        p = Path("data") / self.cfg.data_type / self.cfg.period
        flist = [f for f in p.glob("*") if self.cfg.ticker in f.stem]
        if len(flist) == 1:
            return {"metatrader": self.metatrader,
                    "FORTS": self.metatrader,
                    "bitfinex": self.bitfinex,
                    "yahoo": self.yahoo,
                    }.get(self.cfg.data_type, None)(flist[0])
        else:
            raise FileNotFoundError(p)
    
    def _trim_by_date(self, hist):
        if self.cfg.date_start is not None:
            date_start = pd.to_datetime(self.cfg.date_start, utc=True)
            for i, d in enumerate(hist.Date):
                if d >= date_start:
                    break
            hist = hist.iloc[i:]   
            
        if self.cfg.date_end is not None:
            date_end = pd.to_datetime(self.cfg.date_end, utc=True)
            for i, d in enumerate(hist.Date):
                if d >= date_end:
                    break
            hist = hist.iloc[:i]   
        return hist           
        
    def metatrader(self, data_file):
        pd.options.mode.chained_assignment = None
        hist = pd.read_csv(data_file, sep="\t")
        hist.columns = map(lambda x:x[1:-1], hist.columns)
        hist.columns = map(str.capitalize, hist.columns)
        if "Time" not in hist.columns:
            hist["Time"] = ["00:00:00"]*hist.shape[0]
        hist["Date"] = pd.to_datetime([" ".join([d, t]) for d, t in zip(hist.Date.values, hist.Time.values)])#, utc=True)
        hist = self._trim_by_date(hist)
        hist.drop("Time", axis=1, inplace=True)
        columns = list(hist.columns)
        columns[-2] = "Volume"
        hist.columns = columns
        hist["Id"] = list(range(hist.shape[0]))
        hist_dict = EasyDict({c:hist[c].values for c in hist.columns})
        # hist_dict["Date"] = hist["Date"].values
        hist.set_index("Date", inplace=True, drop=True)
        return hist, hist_dict
    
    def bitfinex(self, data_file):
        hist = pd.read_csv(data_file, header=1)
        hist = hist[::-1]
        hist["Date"] = pd.to_datetime(hist.date.values)
        hist = self._trim_by_date(hist)
        hist.set_index("Date", inplace=True, drop=False)
        hist["Id"] = list(range(hist.shape[0]))
        hist.drop(["unix", "symbol", "date"], axis=1, inplace=True)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Volume"] = hist.iloc[:, -3]
        hist_dict = EasyDict({c:hist[c].values for c in hist.columns})
        return hist, hist_dict
    
    def yahoo(self, data_file):
        hist = pd.read_csv(data_file)
        hist["Date"] = pd.to_datetime(hist.Date, utc=True)
        hist = self._trim_by_date(hist)
        hist.set_index("Date", inplace=True, drop=False)
        hist["Id"] = list(range(hist.shape[0]))
        hist.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
        hist.columns = map(str.capitalize, hist.columns)
        hist["Volume"] = hist.iloc[:, -3]
        hist_dict = EasyDict({c:hist[c].values for c in hist.columns})
        return hist, hist_dict
        

def collect_train_data(dir, fsize=64):
    cfgs, btests = [], []
    for p in sorted(Path(dir).glob("*.pickle")):
        cfg, btest = pickle.load(open(p, "rb"))
        cfgs.append(cfg)
        btests.append(btest)
    print(len(btests))

    tfdict = {"M5":0, "M15":1, "H1":2, "D":3}
    X, y = [], []
    for btest in tqdm(btests, "Load pickles"):
        # print(btest.cfg.ticker, end=" ")
        hist_pd, hist = DataParser(btest.cfg).load()
        mw = MovingWindow(hist, fsize+2)
        # print(len(btest.positions))
        for pos in btest.positions[4:]:
            f, _ = mw(pos.open_indx)
            x = build_features(f, 
                            pos.dir, 
                            btest.cfg.stops_processor.func.cfg.sl, 
                            btest.cfg.trailing_stop_rate,
                            pos.open_date, 
                            tfdict[btest.cfg.period])
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

    tfdict = {"M5":0, "M15":1, "H1":2, "D":3}
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
                posdict[k] = {"sl":[sl], "prof":[profit], "dir":[dir], "id":[pos_id]}
    

    btest = btests[0]
    hist_pd, hist = DataParser(btest.cfg).load()
    mw = MovingWindow(hist, fsize)
    for open_date, pos in posdict.items():
        if len(set(pos["sl"])) == nparams and len(set(pos["dir"])) == 1:
            f, _ = mw(pos["id"][0])
            x = build_features(f, 
                            pos["dir"][0],
                            0, 
                            btest.cfg.trailing_stop_rate,
                            open_date, 
                            tfdict[btest.cfg.period])
            X.append([x])
            y.append(pos["prof"])
    
    X, y = np.array(X), np.array(y, dtype=np.float32)
    print(X.shape, y.shape)
    print(f"{X[0, 0, -2, 0]:8.0f} -> {X[-1, 0, -2, 0]:8.0f}")
    return X, y


class MovingWindow():
    def __init__(self, hist, size):
        self.hist = hist
        self.size = size
        self.data = EasyDict(Date=np.empty(size, dtype=np.datetime64),
                             Id=np.zeros(size, dtype=np.int64),
                             Open=np.zeros(size, dtype=np.float32),
                             Close=np.zeros(size, dtype=np.float32),
                             High=np.zeros(size, dtype=np.float32),
                             Low=np.zeros(size, dtype=np.float32),
                             Volume=np.zeros(size, dtype=np.int64)
                             )
        
    def __call__(self, t):
        t0 = perf_counter()
        self.data.Date = self.hist.Date[t-self.size+1:t+1]
        self.data.Id[:] = self.hist.Id[t-self.size+1:t+1]
        self.data.Open[:] = self.hist.Open[t-self.size+1:t+1]
        self.data.Close[:-1] = self.hist.Close[t-self.size+1:t]
        self.data.Close[-1] = self.data.Open[-1]
        self.data.High[:-1] = self.hist.High[t-self.size+1:t]
        self.data.High[-1] = self.data.Open[-1]
        self.data.Low[:-1] = self.hist.Low[t-self.size+1:t]
        self.data.Low[-1] = self.data.Open[-1]
        self.data.Volume[:-1] = self.hist.Volume[t-self.size+1:t]
        self.data.Volume[-1] = 0      
        return self.data, perf_counter() - t0
    
    
if __name__ == "__main__":
    X, y = collect_train_data2("./optimization")
    get_data(X, y)