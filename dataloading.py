import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def build_features(f, dir, sl, trailing_stop_rate, open_date=None, timeframe=None):
    fo = f.Open[:-2]/f.Open[-2]
    fc = f.Close[:-2]/f.Open[:-2]
    fh = f.High[:-2]/f.Open[:-2]
    fl = f.Low[:-2]/f.Open[:-2]
    fv = f.Volume[:-2]/f.Volume[-2] if f.Volume[-2] != 0 else np.ones_like(f.Volume[:-2])

    if dir > 0:
        x = np.vstack([fc, fo, fl, fh])
    else:
        x = np.vstack([2-fc, 2-fo, 2-fl, 2-fh])
    x = x*100 - 100
    x = np.vstack([x, fv])
    x = np.vstack([x, np.ones(x.shape[1])*sl/6+1])
    x = np.vstack([x, np.ones(x.shape[1])*trailing_stop_rate/0.04+1])
    if open_date is not None:
        odate = pd.to_datetime(open_date)
        odate = odate.year*10000 + odate.month*100 + odate.day        
        x = np.vstack([x, np.ones(x.shape[1])*odate])
    if timeframe is not None:
        x = np.vstack([x, np.ones(x.shape[1])*timeframe])
    return x


def get_data(X, y, test_split=0.25):
    ids = np.arange(X.shape[0])
    # np.random.shuffle(ids)
    test_size = int(X.shape[0]*test_split)
    ids_test, odates_testset, odates = [], set(), X[:, 0, -2, 0]
    while len(ids_test) < test_size:
        ix = np.random.randint(0, X.shape[0])
        d = odates[ix]
        if d not in odates_testset:
            ii = ids[odates == d]
            ids_test += ii.tolist()
            odates_testset.add(d)
    ids_train = [ix for ix in ids if ix not in ids_test]   
    np.random.shuffle(ids_train) 
    np.random.shuffle(ids_test) 
        
    # ids_test, ids_train = ids[:test_size], ids[test_size:]
    X_train, X_test, y_train, y_test, profs_test = X[ids_train], X[ids_test], y[ids_train], y[ids_test], y[ids_test]
    tf_test = X_test[:, 0, -1, 0]
    X_train = X_train[:, :, :-2, :]
    X_test = X_test[:, :, :-2, :]
    
    y_train = np.tanh(y_train)
    y_test = np.tanh(y_test)
    
    return X_train, X_test, y_train, y_test, profs_test, tf_test

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