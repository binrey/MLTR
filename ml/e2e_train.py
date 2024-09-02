from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from data_processing.dataloading import DataParser, MovingWindow
from data_processing.train_dataset import next_price_prediction
from ml.models import E2EModel, E2EModelConv, SeqOutput
from utils import PyConfig, FeeRate



def autoregress_sequense(model, p, features, output_sequense=False, device="cpu"):
    if type(p) is np.ndarray:
        p = torch.from_numpy(p).to(device)
    if type(features) is np.ndarray:
        features = torch.from_numpy(features).to(device)
    dp = p[1:] - p[:-1]
    output_seq, result_seq, fee_seq = np.zeros(
        dp.shape[0]+1), np.zeros(dp.shape[0]+1), np.zeros(dp.shape[0]+1)
    profit = torch.zeros((1, 1), device=device)
    pred_result = torch.zeros((1, 1), device=device)
    output = torch.zeros((1, 1, 1), device=device)
    output_last = torch.zeros((1, 1, 1), device=device)
    pred_result = torch.zeros((1, 1, 1), device=device)
    for i in range(dp.shape[0]):
        # print(f"t={i + 1:04}", end=" ")
        output = model(features[i:i+1])
        fees = (output - output_last).abs() * p[i] * 0.001
        pred_result = dp[i] * output - fees
        output_last = output
        if output_sequense:
            output_seq[i+1] = output.item()
            result_seq[i+1] = pred_result.item()
            fee_seq[i+1] = fees.item()
        else:
            # print(f"{epoch + 1:03} {i + 1:04}: profit += {output.item():7.2f} * {dp[i]:7.2f} - {fees.item():7.3f} = {pred_result.item():7.2f}", end=" ")
            profit += pred_result.squeeze()

            # print(f"| profit: {profit.item():9.3f}")
    if output_sequense:
        return output_seq, result_seq, fee_seq
    else:
        return profit

def batch_sequense(model, p, features, fee_rate:FeeRate, device="cpu") -> SeqOutput:
    if type(p) is np.ndarray:
        p = torch.from_numpy(p).to(device)
    if type(features) is np.ndarray:
        features = torch.from_numpy(features).float().to(device)
    output = model(features).squeeze()
    return SeqOutput(model_ans=output,
                     price=p,
                     fees=fee_rate.order_execution_fee(p[:-1], (output[1:] - output[:-1]).abs()))


class TrainSet:
    def __init__(self, features: np.ndarray, price: np.ndarray, val_size=0):
        self.features = features
        self.price = price
        self.val_size = val_size
        self.val_indx = max(int(self.val_size * self.features.shape[0]), 1)
    
    @property
    def features_train(self):
        return self.features[: -self.val_indx]
    
    @property
    def price_train(self):
        return self.price[: -self.val_indx]
    
    @property
    def features_val(self):
        return self.features[-self.val_indx :]
    
    @property
    def price_val(self):
        return self.price[-self.val_indx :]
    

class E2ETrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tb_writer = None
        self.ticker = self.cfg.ticker
        self.train_sets: Dict[TrainSet] = {}
        self.val_train = None

    def init_model(self):
        n_indics, n_feats = self.train_sets[self.ticker].features.shape[-2:]
        self.model = E2EModel(n_indicators=n_indics, 
                              n_features=n_feats, 
                              nh=8, cls_head=False)
        logger.debug("\n" + str(summary(self.model, [1, n_indics, n_feats], device="cpu")))

    def load_data(self, dataset_root="data", max_size=np.Inf, val_size=0):
        self.val_size = val_size
        mw = MovingWindow(DataParser(self.cfg).load(dataset_root)[1], self.cfg)
        self.train_sets[self.ticker] = TrainSet(
            *next_price_prediction(mw, self.cfg.body_classifier.func, max_size + 1),
            val_size=val_size
        )
        for ticker in self.cfg.add_tickers_to_train:
            cfg = copy(self.cfg)
            cfg.ticker = ticker
            mw = MovingWindow(DataParser(cfg).load(dataset_root)[1], cfg)
            self.train_sets[ticker] = TrainSet(
                *next_price_prediction(mw, self.cfg.body_classifier.func, max_size + 1),
                val_size=val_size
            )
        
    def compute_hold(self, p=None, hold_sign=None, output_last=False, norm=False):
        if p is None:
            p = self.p
        if hold_sign is None:
            if type(p) is torch.Tensor:
                hold_sign = (p[-1] - p[0]).sign()
            else:
                hold_sign = np.sign(p[-1] - p[0])
        hold = (p[1:] - p[:-1]) * hold_sign
        if norm:
            hold /= p[:-1]
        return hold[-1] if output_last else np.hstack([np.array([0]), hold.cumsum()]), hold_sign

    @staticmethod
    def calculate_loss(seq_result:SeqOutput, hold_train, p0, activity_reduction_factor=0.9, ticker_factor=1):
        act_mult = (1 - activity_reduction_factor * seq_result.model_ans.abs().sum()/seq_result.model_ans.shape[0])
        loss = seq_result.sum_profit_relative() * act_mult
        return loss * ticker_factor
        
    @staticmethod
    def array2tensor(np_array, device):
        return torch.from_numpy(np_array.astype(np.float32)).to(device)

    def train(
        self, num_epochs, resume=False, device="cpu", run_name="default"
    ):

        train_tickers = list(self.train_sets.keys())
        
        hold_train, hold_sign = self.compute_hold(self.train_sets[self.ticker].price_train, 
                                                  output_last=True)
        hold_val, _ = self.compute_hold(self.train_sets[self.ticker].price_val, 
                                        hold_sign, 
                                        output_last=True)

        start_epoch = 1
        if not resume:
            self.init_model()
            self.tb_writer = SummaryWriter()
        else:
            assert self.model is not None, "Нет модельки, поставь resume=False"
            start_epoch = self.model.train_info["last_epoch"] + 1
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), maximize=True, lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        # ticker_factors, hold_sum = {}, 0
        # for ticker in train_tickers:
        #     hold, _ = self.compute_hold(self.train_sets[ticker].price_train, output_last=True, norm=True)
        #     ticker_factors[ticker] = hold / hold_train
        #     hold_sum += hold
        ticker_factors = {"SBER": 1, "LKOH": 1, "GAZP": 1}
        # Training loop
        pbar = tqdm(start_epoch + np.arange(num_epochs), desc="Training")
        for epoch in pbar:
            optimizer.zero_grad()
            loss = 0
            for ticker in train_tickers:
                features = self.array2tensor(self.train_sets[ticker].features_train, device=device)
                price = self.array2tensor(self.train_sets[ticker].price_train, device=device)
                seq_result = batch_sequense(
                    self.model, price, features, self.cfg.fee_rate, device=device
                )
                # hold_train, _ = self.compute_hold(price, output_last=True, norm=True)
                loss += self.calculate_loss(seq_result, None, None, 1, ticker_factors.get(ticker, 1))
                
            loss /= len(train_tickers)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.model.train_info["last_epoch"] = epoch
            self.tb_writer.add_scalar("loss/train", loss, epoch)
            self.tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            if (epoch - 1) % 10 == 0:
                with torch.no_grad():
                    features = self.array2tensor(self.train_sets[self.ticker].features_val, device=device)
                    price = self.array2tensor(self.train_sets[self.ticker].price_val, device=device)
                    self.model.eval()
                    seq_result = batch_sequense(
                        self.model,
                        price,
                        features,
                        self.cfg.fee_rate,
                        device=device,
                    )
                    loss_val = seq_result.sum_profit_relative()# - hold_val
                self.tb_writer.add_scalar("loss/val", loss_val, epoch)

            pbar.set_postfix({"loss train": loss.item(), "val": loss_val.item()})

        self.tb_writer.flush()
        return self.model


if __name__ == "__main__":
    DEVICE = torch.device("cpu")

    cfg = PyConfig("zz.py").test()

    model = train(cfg, num_epochs=3)

    model.eval()
    output_seq, result_seq, fee_seq = autoregress_sequense(
        model, p, features, output_sequense=True, device=DEVICE
    )

    fig, ax1 = plt.subplots()
    # ax1.plot(p - p[0], linewidth=3)
    ax1.plot(result_seq.cumsum(0))
    ax1.plot(fee_seq.cumsum(0))
    ax1.plot(hold.to("cpu"), linewidth=3)

    ax2 = ax1.twinx()
    ax2.bar(list(range(output_seq.shape[0])), height=output_seq, width=1, alpha=0.4)

    plt.legend(["prediction", "fees", "buy&hold"])
    plt.tight_layout()
    plt.grid("on")
    plt.show()
