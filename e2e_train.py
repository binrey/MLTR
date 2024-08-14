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
from ml.models import E2EModel, autoregress_sequense, batch_sequense
from utils import PyConfig


class E2ETrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tb_writer = None

    def init_model(self):
        inp_shape = (1, self.features.shape[-1])
        self.model = E2EModel(inp_shape, 8)
        logger.debug("\n" + str(summary(self.model, [inp_shape], device="cpu")))

    def load_data(self, dataset_root="data", max_size=np.Inf):
        mw = MovingWindow(DataParser(self.cfg).load(dataset_root)[1], self.cfg)
        self.features, self.p = next_price_prediction(
            mw, self.cfg.body_classifier.func, self.cfg.hist_buffer_size, max_size + 1
        )

    def compute_hold(self, p=None):
        if p is None:
            p = self.p
        hold = p if p[-1] - p[0] >= 0 else -p
        return hold - hold[0]

    def train_val_split(self, val_size):
        features_val = self.features[-int(val_size * self.features.shape[0]) :]
        p_val = self.p[-int(val_size * self.p.shape[0]) :]
        features_train = self.features[: -int(val_size * self.features.shape[0])]
        p_train = self.p[: -int(val_size * self.p.shape[0])]
        return features_train, p_train, features_val, p_val

    def train(
        self, num_epochs, resume=False, device="cpu", val_size=0.2, run_name="default"
    ):
        features_train, p_train, features_val, p_val = self.train_val_split(val_size)
        features_train = torch.from_numpy(features_train.astype(np.float32)).to(device)
        p_train = torch.from_numpy(p_train.astype(np.float32)).to(device)
        features_val = torch.from_numpy(features_val.astype(np.float32)).to(device)
        p_val = torch.from_numpy(p_val.astype(np.float32)).to(device)

        hold_train = self.compute_hold(p_train)[-1]
        hold_val = self.compute_hold(p_val)[-1]

        start_epoch = 1
        if not resume:
            self.init_model()
            self.tb_writer = SummaryWriter()
        else:
            assert self.model is not None, "Нет модельки, поставь resume=False"
            start_epoch = self.model.train_info["last_epoch"] + 1
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), maximize=True, lr=0.005)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        # Training loop
        pbar = tqdm(start_epoch + np.arange(num_epochs), desc="Training")
        for epoch in pbar:
            optimizer.zero_grad()
            loss = batch_sequense(self.model, p_train, features_train, device=device)
            loss -= hold_train
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.model.train_info["last_epoch"] = epoch
            self.tb_writer.add_scalar("loss/train", loss, epoch)
            self.tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            if (epoch - 1) % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    loss_val = autoregress_sequense(
                        self.model, p_val, features_val, device=device
                    )
                    loss_val -= hold_val
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
