from .base import *


class ClsZigZag(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsZigZag, self).__init__(cfg, name="zigzag")
        self.zigzag = ZigZagNew(self.cfg.period, self.cfg.feature_size)
        self.last_zz_type = 0

    def getfeatures(self):
        zz_values = self.zigzag.values
        features = np.zeros(self.cfg.feature_size)
        features[-zz_values.shape[0]:] = zz_values
        return features
    
    def check_status(self, h):
        zz_ids, zz_values, zz_types = self.zigzag.update(h)
        if self.last_zz_type != zz_types[-1]:
            self.last_zz_type = zz_types[-1]
            features = self.getfeatures    
            return True
        return False   
    
    def __call__(self, common, h) -> bool:
        is_fig = False
        if self.check_status(h):
            features = self.getfeatures
            is_fig = True
        
        if is_fig:
            self.last_zz_type = self.zigzag.types[-1]
            common.lines = [[(x, y) for x, y in zip(self.zigzag.ids, self.zigzag.values)]]
            common.lprice = h.Open[-1] if self.zigzag.types[-1] == 1 else None
            common.sprice = h.Open[-1] if self.zigzag.types[-1] == -1 else None
            # common.sl = {1: min(peaks_bottom), -1: max(peaks_upper)}  
            # common.tp = {1: tp, -1: tp}
            # common.cprice = {-1: min(zz_values[-last_id:]), 1: max(zz_values[-last_id:])} 
        return is_fig