import sys
from experts import ExpertFormation, PyConfig
from backtest import backtest
from pathlib import Path
from dataloading import get_data, build_features, collect_train_data
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from ml import train
import matplotlib.pyplot as plt
logger.remove()
logger.add(sys.stderr, level="INFO")


if __name__ == "__main__":
    test_split_size = 0.2
    device = "cuda"
    cfg = PyConfig().test()
    cfg.date_start="2022-08-01"
    cfg.date_end="2024-01-01"
    cfg.run_model_device = device
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    X, y = collect_train_data("./optimization")
    for i in range(5):
        X_train, X_test, y_train, y_test, profs_train, profs_test, tf_test, _ = get_data(X, y, test_split_size, 4, 5)
        X_train = torch.tensor(X_train).float().to(device)
        model = train(X_train, y_train, None, None, batch_size=512, epochs=1, device=device, calc_test=False)
        model.eval()
        X_train = X_train.float().to(device)
        p_train = model(X_train).detach().cpu().numpy().squeeze()[:, 0]    
        profsum_best, threshold = -999999, np.percentile(p_train, 20)
        # for th in np.arange(0., 0.9, 0.05):
        #     w_profs_train = (p_train > th).astype(np.float32)
        #     profsum = (profs_train*w_profs_train).sum()
        #     if profsum > profsum_best:
        #         profsum_best = profsum
        #         threshold = th
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
    
