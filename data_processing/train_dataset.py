from time import time

import numpy as np
from loguru import logger

from data_processing.dataloading import MovingWindow
from utils import cache_result


# @cache_result
def next_price_prediction(mw: MovingWindow, classifier, max_size=100):
    t0 = time()
    p, features = [], []
    for hist_window in mw(output_time=False):
        if classifier.check_status(hist_window):
            p.append(hist_window.Open[-1])
            features.append(classifier.getfeatures(hist_window))
            if len(p) >= max_size - 1:
                break

    features = np.array(features)
    if features.ndim == 2:
        features = np.expand_dims(features, 1).astype(np.float32)
    p = np.array(p).astype(np.float32)
    logger.info(
        f"Generation completed in {time()-t0:.1f} sec, features: {features.shape}, prices: {p.shape}")

    return features, p
