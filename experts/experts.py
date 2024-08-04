from abc import ABC, abstractmethod
from time import perf_counter
from copy import deepcopy
import numpy as np
import yaml
from easydict import EasyDict
from loguru import logger
from pathlib import Path
import pandas as pd
from collections import OrderedDict

from indicators import *
from backtest_broker import Order
# import torch
from backtest_broker import Broker, Position
from .base import *

        
class ClsTrend(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTrend, self).__init__(cfg, name="trend")
        # self.zigzag = ZigZagOpt(max_drop=self.cfg.maxdrop)
        self.zigzag = ZigZagNew(self.cfg.period)
        
    def __call__(self, common, h) -> bool:
        ids, values, types = self.zigzag.update(h)
        is_fig = False
        if len(ids) >= self.cfg.npairs*2+1:
            flag = False
            if types[-2] > 0:
                flag = values[-2] > values[-4] and values[-3] > values[-5]
                if self.cfg.npairs == 3:
                    flag = flag and values[-4] > values[-6] and values[-5] > values[-7]
            if types[-2] < 0:
                flag = values[-2] < values[-4] and values[-3] < values[-5]
                if self.cfg.npairs == 3:
                    flag = flag and values[-4] < values[-6] and values[-5] < values[-7]
            if flag:
                is_fig = True
                trend_type = types[-2]                        
    
        if is_fig:
            i = self.cfg.npairs*2 + 1
            common.sl = {1: values[-3], -1: values[-3]} 
            common.lines = [[(x, y) for x, y in zip(ids[-i:-1], values[-i:-1])]]
            common.lprice = values[-1] if trend_type > 0 else None
            common.sprice = values[-1] if trend_type < 0 else None
            common.cprice = common.lines[0][-2][1]
        return is_fig


class ClsTunnel(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTunnel, self).__init__(cfg, name="tunnel")
        
    def __call__(self, common, h) -> bool:
        is_fig = False
        best_params = {
            "metric": 0,
            "i": 0,
            "line_above": None,
            "line_below": None,
        }
        for i in range(4, h.Id.shape[0], 1):
            line_above = h.High[-i:].mean()
            line_below = h.Low[-i:].mean()
            middle_line = (line_above + line_below) / 2
            
            if h.Close[-1] < line_above and h.Close[-1] > line_below:
                metric = i / ((line_above - line_below) / middle_line) / 100
                                        
                if metric > best_params["metric"]:
                    best_params.update(
                        {"metric": metric,
                        "i": i,
                        "line_above": line_above,
                        "line_below": line_below,
                        "middle_line": middle_line
                        }
                    )                   
                    
        if best_params["metric"] > self.cfg.ncross:
            is_fig = True
            # break

        if is_fig:
            i = best_params["i"]
            common.sl = {1: h.Low[-i:].min(), -1: h.High[-i:].max()}   
            # v1
            common.lprice = best_params["line_above"]
            common.sprice = best_params["line_below"] 
            # v2
            # if middle_line > h.Close.mean():
            #     common.lprice = line_below
            # else:
            #     common.sprice = line_above
            common.lines = [[(h.Id[-i], best_params["line_above"]), (h.Id[-1], best_params["line_above"])], 
                            [(h.Id[-i], best_params["line_below"]), (h.Id[-1], best_params["line_below"])],
                            [(h.Id[-i], best_params["middle_line"]), (h.Id[-1], best_params["middle_line"])]]

        return is_fig


class ClsTriangle(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTriangle, self).__init__(cfg, name="trngl_simp")
        self.zigzag = ZigZagOpt(max_drop=0.1)
        #self.zigzag = ZigZag2()
        
    def __call__(self, common, h) -> bool:
        ids, values, types = self.zigzag.update(h)        
        is_fig = False
        if len(ids) > self.cfg.npairs*2+1:
            flag2, flag3 = False, False
            if types[-2] > 0:
                flag2 = values[-2] < values[-4] and values[-3] > values[-5]
                if self.cfg.npairs == 3:
                    flag3 = values[-4] < values[-6] and values[-5] > values[-7]
            if types[-2] < 0:
                flag2 = values[-2] > values[-4] and values[-3] < values[-5]
                if self.cfg.npairs == 3:
                    flag3 = values[-4] > values[-6]  and values[-5] < values[-7]
            if (self.cfg.npairs <= 2 and flag2) or (self.cfg.npairs == 3 and flag2 and flag3):
                is_fig = True
                    
        if is_fig:
            i = self.cfg.npairs*2 + 1
            common.lines = [[(x, y) for x, y in zip(ids[-i:-1], values[-i:-1])]]
            common.lprice = max(common.lines[0][-3][1], common.lines[0][-2][1])
            common.sprice = min(common.lines[0][-3][1], common.lines[0][-2][1]) 
            common.sl = {1: min(common.lines[0][-3][1], common.lines[0][-4][1]), 
                        -1: max(common.lines[0][-3][1], common.lines[0][-4][1])} 
            common.tp = {1: common.lprice + abs(common.lprice - common.sl[1])*5, 
                        -1: common.sprice - abs(common.sprice - common.sl[-1])*5} 
        return is_fig


class ClsBB(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsBB, self).__init__(cfg, name="BB")
        
    def _bolinger_beams(self, h):
        mean = h.Close.mean()
        std = h.Close.std()
        return mean, mean + 1*std, mean - 1*std
        
    def __call__(self, common, h) -> bool:
        dir = 0
        mean, bb_high, bb_low = self._bolinger_beams(h)   
        if h.Close[-3] > bb_low and h.Close[-2] < bb_low:
            dir = 1
        if h.Close[-3] < bb_high and h.Close[-2] > bb_high:
            dir = -1            
                 
        if dir != 0:
            common.lines = [[(h.Id[-3], bb_low), (h.Id[-2], bb_low)], [(h.Id[-3], bb_high), (h.Id[-2], bb_high)]]
            if dir > 0:
                common.lprice = h.Open[-1]
            if dir < 0:
                common.sprice = h.Open[-1]
            common.sl = {1: h.Low[-10:].min(), -1: h.High[-10:].max()} 
            common.tp = {1: h.Close[-1] + 2*abs(h.Close[-1] - common.sl[1]), 
                        -1: h.Close[-1] - 2*abs(h.Close[-1] - common.sl[-1])
                        } 
        return dir != 0


class ClsLevels(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsLevels, self).__init__(cfg, name="levels")
        self.ma = {}
        self.n_extrems = self.cfg.n_extrems
        self.show_n_peaks = min(self.cfg.show_n_peaks, self.cfg.n_extrems)
        self.extrems = {"ids": np.zeros(self.n_extrems, dtype=np.int32), "values": np.zeros(self.n_extrems), "sides": np.zeros(self.n_extrems)}
        self.last_cross = 0
        self.cur_cross = 0
        self.last_extr = (None, None)
        self.tmp_extr = (None, None, 0)
        self.last_n = self.cfg.n
        self.active_level = {"extr": None, "dir": 0}
        
    
    def _last_extrem_side(self):
        side = 0
        if len(self.extrems) > 0:
            last_id = max(self.extrems.keys())
            side = 1 if self.extrems[last_id] > self.ma[last_id] else -1
        return side
    
    def _upd_peaks(self, id, val, side):
        # shift old values from right to left and assign last ids, values, sides with new values
        self.extrems["ids"] = np.roll(self.extrems["ids"], -1)
        self.extrems["ids"][-1] = id
        self.extrems["values"] = np.roll(self.extrems["values"], -1)
        self.extrems["values"][-1] = val
        self.extrems["sides"] = np.roll(self.extrems["sides"], -1)
        self.extrems["sides"][-1] = side
        
    def _update_active_level(self, extr: tuple, type="extr"):
        if self.active_level[type] is None:
            self.active_level[type] = extr
        else:
            self.active_level[type] = (min(self.active_level[type][0], extr[0]), 
                                        (self.active_level[type][1] + extr[1])/2)
    
    def check_level_cross(self, h, side, cur_extr_val):
        s = sum((side*(h.Close[-self.cfg.ncross-1:-1] - cur_extr_val)) >= 0)
        return s
    
    def update_inner_state(self, h):     
        id_cur = h.Id[-1]
        self.ma[id_cur] = h.Close[-self.cfg.ma:-1].mean() 
        self.last_cross = self.cur_cross
        if len(self.ma) > 2:
            if h.Close[-2] > self.ma[id_cur]:
                self.cur_cross = 1   
                if h.Close[-3] <= self.ma[id_cur-1]:
                    if self.last_n > self.cfg.n and self.last_extr[0] is not None and self.extrems["sides"][-1] <= 0:    
                        self._upd_peaks(self.last_extr[0]-1, self.last_extr[1], self.cur_cross)         
                    self.last_extr = (id_cur, h.High[-2])  
                    self.last_n = 0
                    
            if h.Close[-2] < self.ma[id_cur]:
                self.cur_cross = -1
                if h.Close[-3] >= self.ma[id_cur-1]:
                    if self.last_n > self.cfg.n and self.last_extr[0] is not None and self.extrems["sides"][-1] >= 0:              
                        self._upd_peaks(self.last_extr[0]-1, self.last_extr[1], self.cur_cross)         
                    self.last_extr = (id_cur, h.Low[-2])
                    self.last_n = 0
        self.last_n += 1            
              
        if self.cur_cross > 0 and self.last_extr[1] is not None and h.High[-2] > self.last_extr[1]:
            self.last_extr = (id_cur, h.High[-2])
        if self.cur_cross < 0 and self.last_extr[1] is not None and h.Low[-2] < self.last_extr[1]:
            self.last_extr = (id_cur, h.Low[-2])
                 
        self.active_level = {"extr": None, "dir": 0}
        extrs2del = []
        for i_extr in range(self.n_extrems):
            if self.extrems["sides"][i_extr] == 0:
                continue                         
            cur_extr_val = self.extrems["values"][i_extr]
            cur_extr_id = self.extrems["ids"][i_extr]
            s = h.Close[-self.cfg.ncross-1:-1] - cur_extr_val
            if sum(s >= 0) == self.cfg.ncross: #h.Close[-2] > cur_extr_val:#
                if h.Close[-self.cfg.ncross-2] < cur_extr_val:
                    self._update_active_level((cur_extr_id, cur_extr_val))
                    self.active_level["dir"] = 1
                    extrs2del.append(i_extr)
                # elif h.Low[-self.cfg.ncross-1] <= cur_extr_val and h.Open[-self.cfg.ncross-1] >= cur_extr_val and self.cur_cross < 0:
                #     self._update_active_level((cur_extr_id, cur_extr_val))
                #     self.active_level["dir"] = 1
            if sum(s <= 0) == self.cfg.ncross: #h.Close[-2] < cur_extr_val:#
                if h.Close[-self.cfg.ncross-2] > cur_extr_val:
                    self._update_active_level((cur_extr_id, cur_extr_val))
                    self.active_level["dir"] = -1      
                    extrs2del.append(i_extr)    
                # elif h.High[-self.cfg.ncross-1] >= cur_extr_val and h.Open[-self.cfg.ncross-1] <= cur_extr_val and self.cur_cross > 0:
                #     self._update_active_level((cur_extr_id, cur_extr_val))
                #     self.active_level["dir"] = -1
        
        for i_extr in extrs2del:
            for k in self.extrems.keys():
                self.extrems[k][1:i_extr+1] = self.extrems[k][:i_extr]
                self.extrems[k][0] = 0
                
                                
    def __call__(self, common, h) -> bool:
        is_fig = self.active_level["dir"]                                                      
        if is_fig:
            common.lines = [[(t, p) for t, p in self.ma.items()]]
            if sum(abs(self.extrems["sides"])):
                levels = []
                for i_extr in range(self.show_n_peaks):
                    if self.extrems["sides"][-i_extr-1] != 0:
                        levels.append([(self.extrems["ids"][-i_extr-1], self.extrems["values"][-i_extr-1]), 
                                    (h.Id[-2], self.extrems["values"][-i_extr-1])])
                levels.append([(self.active_level["extr"][0], self.active_level["extr"][1]), (h.Id[-2], self.active_level["extr"][1])])
                common.lines += levels
                                                                    
            common.lprice = h.Close[-1] if is_fig > 0 else None
            common.sprice = h.Close[-1] if is_fig < 0 else None
            common.sl = {1: self.active_level["extr"][1], 
                        -1: self.active_level["extr"][1]} 
            # common.tp = {1: common.lprice + abs(common.lprice - common.sl[1])*5, 
            #             -1: common.sprice - abs(common.sprice - common.sl[-1])*5} 
        return is_fig


class ClsBuyAndHold(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsBuyAndHold, self).__init__(cfg, name="buy_and_hold")
        
    def __call__(self, common, h) -> bool:
        raise NotImplementedError
        return False


class ClsDummy(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsDummy, self).__init__(cfg, name="dummy")
        
    def __call__(self, common, h) -> bool:
        common.lprice = h.Close[-2] #max(h.High[-2], h.Low[-2])
        common.sprice = h.Close[-2] #min(h.High[-2], h.Low[-2])
        common.sl = {1: h.Low[-2], -1: h.High[-2]}  
        common.lines = [[(h.Id[-5], common.lprice), (h.Id[-1], common.lprice)], [(h.Id[-5], common.sprice), (h.Id[-1], common.sprice)]]
        return True
        

class ClsCustom(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsCustom, self).__init__(cfg, name="custom")
        folder = "data/handmade_test/AAPL_H1"
        self.signals, self.props, self.features = {}, {}, {}
        for fname in sorted(Path(folder).rglob("*.xlsx")):
            # d = str(fname.stem).split("___")[1]
            df = pd.read_excel(fname)
            side = -1 if "min" in df.Analyze.iloc[1] else 1
            k = np.array(str(df.Date.iloc[0])[:11] + df.Time.iloc[0], dtype='datetime64[ns]').item()
            self.signals[k] = side
            
            self.props[k] = []
            for i in range(df.shape[0]):
                row = df.iloc[i]
                x = np.array(str(row.Date)[:11] + row.Time, dtype="datetime64[ns]")
                y = {"real_peak_max": row.High,
                        "real_peak_min": row.Low}.get(row.Analyze, row.Close)
                self.props[k] += [(x, y)]
                
            df.Date = pd.to_datetime(df.Date)
            dt = np.array([(df.iloc[0].Date - df.iloc[i].Date).days for i in range(0, df.shape[0])])
            dy = [df.iloc[0].Close]
            dy += [df.iloc[i].High if df.iloc[i].Analyze == "real_peak_max" else df.iloc[i].Low for i in range(1, df.shape[0])]
            dy = (np.array(dy) - df.iloc[0].Close)/df.iloc[0].Close*100
            self.features[k] = np.hstack([dt, dy])
        
        from joblib import load
        self.model = load('random_forest_model.joblib')
        
    def __call__(self, common, h) -> bool:
        t = h.Date[-1].astype("datetime64").item()
        side = self.signals.get(t, 0)
        if side != 0:
            prediction = self.model.predict(self.features[t].reshape(1, -1))[0]
            if prediction[0] == 0:
                return False
            if side == 1:
                common.lprice = h.Open[-1] #max(h.High[-2], h.Low[-2])
            if side == -1:
                common.sprice = h.Open[-1] #min(h.High[-2], h.Low[-2])
            common.lines = [[]]    
            for i, pt in enumerate(self.props[t]):
                found_date = h.Id[h.Date == pt[0]]
                if len(found_date):
                    common.lines[0].append((found_date[0], pt[1]))
                if i == 2:
                    common.sl = {1: pt[1], -1: pt[1]} 
                    common.tp = {1: h.Close[-1] + abs(h.Close[-1] - pt[1])*3, 
                                 -1: h.Close[-1] - abs(h.Close[-1] - pt[1])*3} 
            return True
            
        return False
    
    
if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from dataloading import MovingWindow, DataParser
    
    cfg = PyConfig().test()
    expert = ExpertFormation(cfg)
    dp = DataParser(cfg)
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg.hist_buffer_size)
    h, _ = mw(100)
    # x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to("cuda")
    x = torch.tensor(np.zeros((7, 64))).unsqueeze(0).unsqueeze(0).double().to("cuda")
    m = expert.model.double()
    m.eval()
    with torch.no_grad():
        for _ in range(10):
            torch.random.seed = 0
            print(m(x).sum())
    # print(summary(m, (10, 1, 7, 32)))