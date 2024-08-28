import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ExtensionBase


class FileReader(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(FileReader, self).__init__(cfg, name="file_reader")
        source = Path(self.cfg.source_file)
        self.signals, self.props, self.features = {}, {}, {}
        self.signals = pd.read_csv(source, converters={"TIME": pd.to_datetime})
        self.signals.index = self.signals.TIME.values.astype("datetime64")
        self.signals.DIR = self.signals.DIR.map(lambda x: {"UP": 1, "DOWN": -1}[x])
        
        # from joblib import load
        # self.model = load('random_forest_model.joblib')
        
    def __call__(self, common, h) -> bool:
        t = pd.to_datetime(h.Date[-1])
        # Set the new time
        new_time = datetime.time(0, 0, 0)
        t = pd.to_datetime(datetime.datetime.combine(t.date(), new_time))

        side = 0
        if t in self.signals.index:
            row = self.signals.loc[t]
            if row.ndim > 1:
                row = row.iloc[0]
            side = row.DIR
        if side != 0:
            # prediction = self.model.predict(self.features[t].reshape(1, -1))[0]
            # if prediction[0] == 0:
            #     return False
            if side == 1:
                common.lprice = h.Open[-1] #max(h.High[-2], h.Low[-2])
            if side == -1:
                common.sprice = h.Open[-1] #min(h.High[-2], h.Low[-2])   
            common.sl = {1: row.SL, -1: row.SL} 
            return True
            
        return False