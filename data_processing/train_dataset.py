import numpy as np
from backtest import MovingWindow
from loguru import logger


def next_price_prediction(mw: MovingWindow, classifier, hist_buffer_size, max_size=100):
    p, features = [], []
    for hist_window in mw(output_time=False):
        if classifier.check_status(hist_window):
            p.append(hist_window.Open[-1])
            features.append(classifier.getfeatures())
            if len(p) >= max_size - 1:
                break

    features = np.expand_dims(np.array(features), 1).astype(np.float32)
    p = np.array(p).astype(np.float32)
    logger.info(
        f"Generated features shape: {features.shape}, and prices shape: {p.shape}")

    return features, p
