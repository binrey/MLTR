import sys
from experts import ExpertFormation, PyConfig
from backtest import backtest
from pathlib import Path
import pickle
from dataloading import get_data, build_features, DataParser, MovingWindow
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from ml import train
import matplotlib.pyplot as plt
logger.remove()
logger.add(sys.stderr, level="INFO")


def collect_train_data(dir):
    cfgs, btests = [], []
    for p in sorted(Path(dir).glob("*.pickle")):
        cfg, btest = pickle.load(open(p, "rb"))
        cfgs.append(cfg)
        btests.append(btest)
    print(len(btests))

    fsize = 64
    tfdict = {"M5":0, "M15":1, "H1":2}
    X, y, poslist = [], [], []
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
            poslist.append(pos)
            
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    print(X.shape, y.shape)
    print(f"{X[0, 0, -2, 0]:8.0f} -> {X[-1, 0, -2, 0]:8.0f}")
    return X, y


if __name__ == "__main__":
    test_split_size = 0.01
    device = "cuda"
    cfg = PyConfig().test()
    cfg.date_start="2023-01-01"
    cfg.date_end="2024-01-01"
    cfg.run_model_device = device
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    X, y = collect_train_data("./optimization")
    for i in range(10):
        X_train, X_test, y_train, y_test, profs_train, profs_test, tf_test = get_data(X, y, test_split_size)
        X_train = torch.tensor(X_train).float().to(device)
        model = train(X_train, y_train, None, None, batch_size=512, epochs=3, device=device, calc_test=False)
        model.eval()
        X_train = torch.tensor(X_train).float().to(device)
        p_train = model(X_train).detach().cpu().numpy().squeeze()[:, 0]    
        profsum_best, threshold = -999999, 0
        for th in np.arange(0., 0.9, 0.05):
            w_profs_train = (p_train > th).astype(np.float32)
            profsum = (profs_train*w_profs_train).sum()
            if profsum > profsum_best:
                profsum_best = profsum
                threshold = th
        print(threshold)
        model.set_threshold(threshold)
        torch.save(model.state_dict(), "model.pth")
        brok_results = backtest(cfg)
        print(brok_results.profits.sum())
        plt.plot([pos.close_date for pos in brok_results.positions], brok_results.profits.cumsum(), alpha=0.1, color="black")
        plt.grid("on")
        plt.tight_layout()
    
    cfg.run_model_device = None
    brok_results = backtest(cfg)
    print(brok_results.profits.sum())
    plt.plot([pos.close_date for pos in brok_results.positions], brok_results.profits.cumsum(), linewidth=2)
        
    plt.savefig("backtest.png")
    # plt.show()
    
