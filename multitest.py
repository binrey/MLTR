import sys
from experts import PyConfig
from backtest import backtest
from dataloading import get_data, collect_train_data
import numpy as np
from loguru import logger
import torch
from ml import train
import matplotlib.pyplot as plt
logger.remove()
logger.add(sys.stderr, level="INFO")


if __name__ == "__main__":
    test_split_size = 0.25
    device = "mps"
    cfg = PyConfig().test()
    cfg.run_model_device = device
    legend, last_prof = [], 0
    X, y = collect_train_data("./optimization", 32)
    for i in range(int(1/test_split_size)):
        X_train, X_test, y_train, y_test, profs_train, profs_test, tf_test, test_dates = get_data(X, y, test_split_size, i, i+1)
        X_train = torch.tensor(X_train).float().to(device)
        model = train(X_train, y_train, None, None, batch_size=512, epochs=15, device=device, calc_test=False)
        model.eval()
        X_train = X_train.float().to(device)
        p_train = model(X_train).detach().cpu().numpy().squeeze()[:, 0]    
        profsum_best, threshold = -999999, np.percentile(p_train, 10)
        for th in np.arange(0., 0.9, 0.025):
            profsum = f1_score(y_train[:, 0], p_train>th)
            if profsum > profsum_best:
                profsum_best = profsum
                threshold = th
        model.set_threshold(threshold)
        torch.save(model.state_dict(), "model.pth")
        cfg.date_start=f"{test_dates[0][:4]}-{test_dates[0][4:6]}-{test_dates[0][6:]}"
        cfg.date_end=f"{test_dates[1][:4]}-{test_dates[1][4:6]}-{test_dates[1][6:]}"
        brok_results = backtest(cfg)
        cumsum = brok_results.profits.cumsum()
        print(brok_results.profits.sum())
        plt.plot([pos.close_date for pos in brok_results.positions], cumsum + last_prof)
        last_prof += cumsum[-1]
        plt.grid("on")
        plt.tight_layout()
        legend.append(f"{test_dates[0]}-{test_dates[1]}")
    
    cfg.run_model_device = None
    cfg.date_start="2000-01-01"
    cfg.date_end="2024-01-01"
    brok_results = backtest(cfg)
    print(brok_results.profits.sum())
    plt.plot([pos.close_date for pos in brok_results.positions], brok_results.profits.cumsum(), linewidth=2, alpha=0.6)
    plt.legend(legend)
    plt.savefig("backtest.png")
    # plt.show()
    
