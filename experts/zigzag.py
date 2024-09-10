from pathlib import Path

import torch

from .base import *


class ClsZigZag(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsZigZag, self).__init__(cfg, name="zigzag")
        self.zigzag = ZigZag(self.cfg.period, self.cfg.feature_size)
        self.last_zz_type = 0
        self.model = None
        if Path("model.pt").exists():
            self.model = torch.jit.load("model.pt")
            # self.model = torch.load("model.pt")

    def getfeatures(self, h):
        id_mask = self.zigzag.ids != 0
        features = np.zeros((3, self.cfg.feature_size))
        ma = h.Close.mean()
        features[0, id_mask] = (self.zigzag.values[id_mask] - ma)/ma
        features[1, id_mask] = (-self.zigzag.ids[id_mask] + self.zigzag.ids[-1])/h.Close.shape[0]
        volumes_selected = h.Volume[np.in1d(h.Id, self.zigzag.ids)]
        features[2, id_mask] = volumes_selected/volumes_selected.max()
        return features

    def check_status(self, h):
        zz_ids, zz_values, zz_types = self.zigzag.update(h)
        if self.last_zz_type != zz_types[-1]:
            self.last_zz_type = zz_types[-1]
            return True
        return False

    def __call__(self, common, h) -> bool:
        dir = 0
        if self.check_status(h):
            features = self.getfeatures()

            output = self.model(torch.from_numpy(
                features.astype(np.float32).reshape((1, 1, -1)))).item()
            dir = 1 if output > 0 else -1

        if dir != 0:
            self.last_zz_type = self.zigzag.types[-1]
            common.lines = [[(x, y) for x, y in zip(
                self.zigzag.ids, self.zigzag.values)]]
            common.lprice = h.Open[-1] if dir > 0 else None
            common.sprice = h.Open[-1] if dir < 0 else None
            # common.sl = {1: min(peaks_bottom), -1: max(peaks_upper)}
            # common.tp = {1: tp, -1: tp}
            # common.cprice = {-1: min(zz_values[-last_id:]), 1: max(zz_values[-last_id:])}
            return True
        return False
