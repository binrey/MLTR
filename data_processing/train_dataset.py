import torch
from torch import nn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import math
from data_processing.dataloading import DataParser
from easydict import EasyDict
from backtest import MovingWindow
from tqdm import tqdm




def next_price_prediction(cfg):
    data_pd, data_np = DataParser(cfg).load()
    wsize = cfg.hist_buffer_size
    classifier = cfg.body_classifier.func
    fsize = classifier.cfg.feature_size
    
    mw = MovingWindow(data_np, wsize)
    last_type = 0
    p, features = [], []
    for t in range(wsize-1, data_np.Close.shape[0]):
    # for t in range(3000, data_np.Close.shape[0]): # best - 1500
        hist_window = mw(t)[0]
        if classifier.check_status(hist_window):
            p.append(hist_window.Open[-1])
            features.append(classifier.getfeatures())
            if len(p) > 50:
                break
  
    features = np.array(features).astype(np.float32)
    p = np.array(p).astype(np.float32)
    print(features.shape, p.shape)

    return features, p
