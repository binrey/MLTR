import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import PyConfig
from data_processing.train_dataset import next_price_prediction
from data_processing.dataloading import MovingWindow, DataParser
from tqdm import tqdm
from ml.models import autoregress_sequense
from ml.models import E2EModel
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class E2ETrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def init_model(self):
        inp_shape = (1, self.features.shape[1])
        self.model = E2EModel(inp_shape, 16)
        logger.debug("\n" + str(summary(self.model, [inp_shape, (1, self.model.nh)], device="cpu")))

    def load_data(self, max_size=np.Inf):
        mw = MovingWindow(DataParser(self.cfg).load()[1], self.cfg)
        self.features, self.p = next_price_prediction(
            mw, self.cfg.body_classifier.func, self.cfg.hist_buffer_size, max_size+1)

    def compute_hold(self, p=None):
        if p is None:
            p = self.p
        hold = p if p[-1] - p[0] >= 0 else -p
        return hold - hold[0]

    def train(self, num_epochs, resume=False, device="cpu"):
        features = torch.from_numpy(self.features).to(device)
        p = torch.from_numpy(self.p).to(device)
        hold = self.compute_hold(p)
        if not resume:
            self.init_model()
        else:
            assert self.model is not None, "Нет модельки, поставь resume=False"
        self.model.to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), maximize=True, lr=0.001)
        tb_writer = SummaryWriter()

        # Training loop
        pbar = tqdm(list(range(num_epochs)), desc="Training")
        for epoch in pbar:
            optimizer.zero_grad()
            loss, _ = autoregress_sequense(
                self.model, p, features, device=device)
            loss -= hold[-1]
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar("Loss/train", loss, epoch)
            pbar.set_postfix({"loss": loss.item()})

        tb_writer.flush()
        return self.model


if __name__ == "__main__":
    DEVICE = torch.device("cpu")

    cfg = PyConfig("zz.py").test()

    model = train(cfg, num_epochs=3)

    model.eval()
    output_seq, result_seq, fee_seq = autoregress_sequense(
        model, p, features, output_sequense=True, device=DEVICE)

    fig, ax1 = plt.subplots()
    # ax1.plot(p - p[0], linewidth=3)
    ax1.plot(result_seq.cumsum(0))
    ax1.plot(fee_seq.cumsum(0))
    ax1.plot(hold.to("cpu"), linewidth=3)

    ax2 = ax1.twinx()
    ax2.bar(list(range(output_seq.shape[0])),
            height=output_seq, width=1, alpha=0.4)

    plt.legend(["prediction", "fees", "buy&hold"])
    plt.tight_layout()
    plt.grid("on")
    plt.show()
