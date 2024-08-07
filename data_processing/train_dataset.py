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



def next_price_prediction(mw: MovingWindow, classifier, hist_buffer_size, max_size=100):
    last_type = 0
    p, features = [], []
    for hist_window in mw(output_time=False):
        if classifier.check_status(hist_window):
            p.append(hist_window.Open[-1])
            features.append(classifier.getfeatures())
            if len(p) >= max_size - 1:
                break
  
    features = np.array(features).astype(np.float32)
    p = np.array(p).astype(np.float32)
    print(features.shape, p.shape)

    return features, p
