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

from indicators import ZigZag, ZigZag2, ZigZagOpt
from backtest_broker import Order
from dataloading import build_features
import torch
from backtest_broker import Broker


class ExpertBase(ABC):
    def __init__(self):
        self.lines = []
        self.orders = []
            
    @abstractmethod
    def get_body(self) -> None:
        pass
    
    @abstractmethod
    def create_orders(self) -> None:
        pass
    
    def update(self, h, active_position):
        t0 = perf_counter()
        self.active_position = active_position
        self.get_body(h)
        return perf_counter() - t0


class ExtensionBase:
    def __init__(self, cfg, name):
         self.name = name + ":" + "-".join([f"{v}" for k, v in cfg.items()])
         
    def __call__(self, common, h):
        pass
            
    def update_inner_state(self, h):
        pass


class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.trend_maxsize = 1  
        super(ExpertFormation, self).__init__()  
        self.body_cls = cfg.body_classifier.func
        self.stops_processor = cfg.stops_processor.func
        self.wait_length = cfg.wait_entry_point
        self.reset_state()
        self.order_sent = False
        
        if self.cfg.run_model_device is not None:
            from ml import Net, Net2
            self.model = Net2(4, 32)
            self.model.load_state_dict(torch.load("model.pth"))
            # self.model.set_threshold(0.6)
            self.model.eval()
            self.model.to(self.cfg.run_model_device)
        
    def reset_state(self):
        self.order_dir = 0
        self.formation_found = False
        self.wait_entry_point = 0
        self.lprice = None
        self.sprice = None
        self.cprice = None
            
    def get_body(self, h):
        self.body_cls.update_inner_state(h)
        if self.active_position is not None:
            return
        
        self.order_sent = False
        if self.formation_found == False:
            self.formation_found = self.body_cls(self, h)
            if self.formation_found:
                self.wait_entry_point = self.wait_length
            else:
                return
            
        logger.debug(f"{h.Id[-1]} long: {self.lprice}, short: {self.sprice}, cancel: {self.cprice}, open: {h.Open[-1]}")
        
        self.order_dir = 0
        if self.lprice:
            if h.Open[-1] >= self.lprice or h.Close[-2] > self.lprice:
                self.order_dir = 1
            if self.cprice and h.Open[-1] < self.cprice:
                self.reset_state()
                return
        if self.sprice:
            if h.Open[-1] <= self.sprice or h.Close[-2] < self.sprice:
                self.order_dir = -1
            if self.cprice and h.Open[-1] > self.cprice:
                self.reset_state()
                return            
        
        if h.Date[-1] in self.cfg.no_trading_days:
            self.reset_state()
            
        y = None
        if self.cfg.run_model_device and self.order_dir != 0:
            x = build_features(h, 
                               self.order_dir, 
                               self.stops_processor.cfg.sl,
                               self.cfg.trailing_stop_rate)
            x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(self.cfg.run_model_device)
            y = [0.5, 1, 2, 4, 8][self.model.predict(x).item()]
            
        if self.order_dir != 0:
            tp, sl = self.stops_processor(self, h)
            self.create_orders(h.Id[-1], self.order_dir, tp, sl)
            self.order_sent = True
            self.reset_state()

        if self.wait_entry_point == 0:
            self.formation_found = False
        else:
            self.wait_entry_point -= 1

            
class BacktestExpert(ExpertFormation):
    def __init__(self, cfg):
        self.cfg = cfg
        super(BacktestExpert, self).__init__(cfg)
        
    def create_orders(self, time_id, order_dir, tp, sl):
        self.orders = [Order(order_dir, Order.TYPE.MARKET, time_id, time_id)]
        if tp:
            self.orders.append(Order(tp, Order.TYPE.LIMIT, time_id, time_id))
        if sl:
            self.orders.append(Order(sl, Order.TYPE.LIMIT, time_id, time_id))
        logger.debug(f"{time_id} send order {self.orders[0]}, " + 
                        f"tp: {self.orders[1] if len(self.orders)>1 else 'NO'}, " +
                        f"sl: {self.orders[2] if len(self.orders)>2 else 'NO'}")
        
            
class ByBitExpert(ExpertFormation):
    def __init__(self, cfg, session):
        self.cfg = cfg
        self.session = session
        super(ByBitExpert, self).__init__(cfg)
        
    def create_orders(self, time_id, order_dir, tp, sl):
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=self.cfg.ticker,
                side="Buy" if order_dir > 0 else "Sell",
                orderType="Market",
                qty=str(self.cfg.lot),
                timeInForce="GTC",
                # orderLinkId="spot-test-postonly",
                stopLoss="" if sl is None else str(abs(sl)),
                takeProfit="" if tp is None else str(tp)
                )
            logger.debug(resp)
        except Exception as ex:
            logger.error(ex)
        
        
class ClsTrend(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTrend, self).__init__(cfg, name="trend")
        # self.zigzag = ZigZagOpt(max_drop=self.cfg.maxdrop)
        self.zigzag = ZigZag2()
        
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
            # if flag:
                # flag = flag and abs(values[-2] - values[-5])/abs(values[-3] - values[-4]) >= self.cfg.minspace
                # for i in range(2, self.cfg.npairs*2):
                #     flag = flag and ids[-i] - ids[-i-1] >= self.cfg.minspace
                #     if not flag:
                #         break
            if flag:
                is_fig = True
                trend_type = types[-2]                        
    
        if is_fig:
            
            i = self.cfg.npairs*2 + 1
            common.sl = {1: h.Low[-i:].min(), -1: h.High[-i:].max()} 
            common.lines = [[(x, y) for x, y in zip(ids[-i:-1], values[-i:-1])]]
            common.lprice = max(common.lines[0][-1][1], common.lines[0][-2][1]) if trend_type > 0 else None
            common.sprice = min(common.lines[0][-1][1], common.lines[0][-2][1]) if trend_type < 0 else None
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
            # v1
            # line_above = h.Close[-i:].max()
            # line_below = h.Close[-i:].min()
            # v2
            line_above = h.High[-i:].mean()
            line_below = h.Low[-i:].mean()
            # v3
            # line_above = h.High[-i:].max()
            # line_below = h.Low[-i:].min()    
                    
            # v1
            # middle_line = h.Close[-i:].mean()
            # v2
            middle_line = (line_above + line_below) / 2
            
            if h.Close[-1] < line_above and h.Close[-1] > line_below:
                # v1
                metric = i / ((line_above - line_below) / middle_line) / 100

                # v2
                # metric = 0
                # for j in range(i):
                #     if h.High[-j] > middle_line and h.Low[-j] < middle_line:
                #         metric += 1  
                # metric = metric*(1 + 1/i)
                                        
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


class ClsTunnel2(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTunnel2, self).__init__(cfg, name="tuntrend2")
        self.level = None
        
    def __call__(self, common, h) -> bool:
        is_fig = False
        if self.level is None:
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
                #middle_line = h.Close[-i:].mean()
                
                metric = 0
                for j in range(i):
                    if h.High[-j] > middle_line and h.Low[-j] < middle_line:
                        metric += 1  
                # metric = metric*(1 + 1/i)
                                        
                if metric > best_params["metric"]:
                    best_params.update(
                        {"metric": metric, "i": i, "middle_line": middle_line}
                    )                   
                        
            if best_params["metric"] > self.cfg.ncross:
                if h.Close[-2] > line_above:
                    self.level = {"value": best_params["middle_line"], "start_id": h.Id[-i], "end_id": h.Id[-1], "check_line": line_above}
                    
                if h.Close[-2] < line_below:
                    self.level = {"value": best_params["middle_line"], "start_id": h.Id[-i], "end_id": h.Id[-1], "check_line": -line_below}          
            
        if self.level is not None:
            flag = False
            if self.level["check_line"] > 0:
                is_fig = h.Close[-3] < self.level["check_line"]
            if self.level["check_line"] < 0:
                is_fig = h.Close[-3] > self.level["check_line"]

        if is_fig:
            if self.level["check_line"] > 0:
                common.lprice = abs(self.level["check_line"])
                common.cprice = self.level["value"]
            if self.level["check_line"] < 0:
                common.sprice = abs(self.level["check_line"])
                common.cprice = self.level["value"] 
                      
            common.sl = {1: h.Low[-self.level["start_id"]:].min(), -1: h.High[-self.level["start_id"]:].max()}   
            common.lines = [[(self.level["start_id"], self.level["value"]), (self.level["end_id"], self.level["value"])],
                            [(self.level["start_id"], abs(self.level["check_line"])), (h.Id[-1], abs(self.level["check_line"]))]]
            self.level = None
        return is_fig


class ClsTunZigZag(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTunZigZag, self).__init__(cfg, name="tuntrend")
        self.levels = []
        self.zigzag = ZigZag2()
        
    def __call__(self, common, h) -> bool:
        is_fig = False
        best_params = {
            "metric": 0,
            "i": 0,
            "line_above": None,
            "line_below": None,
        }
        for i in range(4, h.Id.shape[0], 1):
            # line_above = h.High[-i:].mean()
            # line_below = h.Low[-i:].mean()
            # middle_line = (line_above + line_below) / 2
            middle_line = h.Close[-i:].mean()
            
            metric = 0
            for j in range(i):
                if h.High[-j] > middle_line and h.Low[-j] < middle_line:
                    metric += 1  
                metric = metric*(1 + 1/i)
                                    
            if metric > best_params["metric"]:
                best_params.update(
                    {"metric": metric,
                    "i": i,
                    "middle_line": middle_line
                    }
                )                   
                    
        if best_params["metric"] > self.cfg.ncross:
            self.levels = [{"value": best_params["middle_line"], "start_id": h.Id[-i]}]
            
        if len(self.levels):
            ids, values, types = self.zigzag.update(h)
            if len(ids) >= 2*2+1:
                flag = False
                if types[-2] > 0 and values[-3] > self.levels[0]["value"]:
                    flag = values[-2] > values[-4] and values[-3] > values[-5]
                if types[-2] < 0 and values[-3] < self.levels[0]["value"]:
                    flag = values[-2] < values[-4] and values[-3] < values[-5]
                if flag:
                    is_fig = True
                    trend_type = types[-2]  

        if is_fig:
            i = best_params["i"]
            common.lines = [[(x, y) for x, y in zip(ids[-5:-1], values[-5:-1])]]
            common.lprice = h.Open[-1] if trend_type > 0 else None
            common.sprice = h.Open[-1] if trend_type < 0 else None
            common.cprice = common.lines[0][-2][1]            
            common.sl = {1: h.Low[-i:].min(), -1: h.High[-i:].max()}   
            common.lines += [[(h.Id[-i], self.levels[0]["value"]), (h.Id[-1], self.levels[0]["value"])]]
            self.levels = []
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
        self.extrems = OrderedDict()
        self.last_cross = 0
        self.cur_cross = 0
        self.last_extr = (None, None)
        self.last_n = self.cfg.n
        self.active_level = {"extr": None, "dir": 0}
        
    def update_inner_state(self, h):
        def remove_last_extrem():
            # remove the last added element of dict self.extrems
            last_key = max(self.extrems.keys())
            self.extrems.pop(last_key) #TODO
            
        def update_active_level(extr: tuple, type="extr"):
            if self.active_level[type] is None:
                self.active_level[type] = extr
            else:
                self.active_level[type] = (min(self.active_level[type][0], extr[0]), 
                                           (self.active_level[type][1] + extr[1])/2)
            
        id_cur = h.Id[-1]
        self.ma[id_cur] = h.Close[-self.cfg.ma:-1].mean() 
        self.last_cross = self.cur_cross
        if len(self.ma) > 2:
            if h.Close[-2] > self.ma[id_cur] and h.Close[-3] <= self.ma[id_cur-1]:
                if self.last_n >= self.cfg.n and self.last_cross != 1:
                    self.cur_cross = 1
                    # if self.last_cross == self.cur_cross:
                    #     remove_last_extrem()
                    if self.last_extr[0] is not None :
                        self.extrems[self.last_extr[0]-1] = self.last_extr[1]
                    self.last_extr = (id_cur, h.High[-2])
                    self.last_n = 0
            if h.Close[-2] < self.ma[id_cur] and h.Close[-3] >= self.ma[id_cur-1]:
                if self.last_n >= self.cfg.n and self.last_cross != -1:
                    self.cur_cross = -1  
                    # if self.last_cross == self.cur_cross:
                    #     remove_last_extrem()
                    if self.last_extr[0] is not None:
                        self.extrems[self.last_extr[0]-1] = self.last_extr[1]
                    self.last_extr = (id_cur, h.Low[-2])
                    self.last_n = 0
              
        if self.cur_cross > 0:
            self.last_n += 1
            if h.High[-2] > self.last_extr[1]:
                self.last_extr = (id_cur, h.High[-2])
        if self.cur_cross < 0:
            self.last_n += 1
            if h.Low[-2] < self.last_extr[1]:
                self.last_extr = (id_cur, h.Low[-2])
                
        self.active_level = {"extr": None, "dir": 0}
        extrs2del = []
        for extr_id in list(self.extrems.keys()):
            cur_extr_val = self.extrems[extr_id]
            if h.Close[-2] > cur_extr_val and h.Close[-3] < cur_extr_val:
                update_active_level((extr_id, cur_extr_val))
                self.active_level["dir"] = 1
                extrs2del.append(extr_id)
            elif h.Close[-2] > cur_extr_val and h.High[-3] <= cur_extr_val:
                update_active_level((extr_id, cur_extr_val))
                self.active_level["dir"] = 1
            if h.Close[-2] < cur_extr_val and h.Close[-3] > cur_extr_val:
                update_active_level((extr_id, cur_extr_val))
                self.active_level["dir"] = -1      
                extrs2del.append(extr_id)    
            elif h.Close[-2] < cur_extr_val and h.Low[-3] >= cur_extr_val:
                update_active_level((extr_id, cur_extr_val))
                self.active_level["dir"] = -1
                
        for extr_id in extrs2del:
            self.extrems.pop(extr_id)
                
                                
                           
    def __call__(self, common, h) -> bool:
        is_fig = self.active_level["dir"]                                                      
        if is_fig:
            common.lines = [[(t, p) for t, p in self.ma.items()]]
            if len(self.extrems):
                levels = []
                for extr_id in list(self.extrems.keys())[-self.cfg.show_n_peaks:]:
                    levels.append([(extr_id, self.extrems[extr_id]), (h.Id[-1], self.extrems[extr_id])])
                levels.append([(self.active_level["extr"][0], self.active_level["extr"][1]), (h.Id[-1], self.active_level["extr"][1])])
                common.lines += levels
                                                                    
            common.lprice = h.Close[-1] if is_fig > 0 else None
            common.sprice = h.Close[-1] if is_fig < 0 else None
            # common.sl = {1: min(common.lines[0][-3][1], common.lines[0][-4][1]), 
            #             -1: max(common.lines[0][-3][1], common.lines[0][-4][1])} 
            # common.tp = {1: common.lprice + abs(common.lprice - common.sl[1])*5, 
            #             -1: common.sprice - abs(common.sprice - common.sl[-1])*5} 
        return is_fig


class ClsDummy(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsDummy, self).__init__(cfg, name="dummy")
        
    def __call__(self, common, h) -> bool:
        common.lprice = h.Close[-2] #max(h.High[-2], h.Low[-2])
        common.sprice = h.Close[-2] #min(h.High[-2], h.Low[-2])
        common.sl = {1: common.sprice, -1: common.lprice}  
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


class StopsFixed(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsFixed, self).__init__(cfg, name="stops_fix")
        
    def __call__(self, common, h, sl_custom=None):
        tp = -common.order_dir*h.Open[-1]*(1+common.order_dir*self.cfg.tp/100) if self.cfg.tp is not None else self.cfg.tp
        sl = self.cfg.sl
        if sl_custom is not None:
            sl = sl_custom
        if sl is not None:
            sl = -common.order_dir*h.Open[-1]*(1-common.order_dir*sl/100)
        return tp, sl
    

class StopsDynamic(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(StopsDynamic, self).__init__(cfg, name="stops_dyn")
        
    def __call__(self, common, h):
        tp, sl = None, None
        if self.cfg.tp_active:
            tp = -common.order_dir*common.tp[common.order_dir]
        if self.cfg.sl_active:
            sl_add = 0#(common.sl[-1] - common.sl[1])
            sl = -common.order_dir*(common.sl[common.order_dir] - common.order_dir*sl_add)
        return tp, sl
    
    
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
    x = build_features(h, 1, 
                       cfg.stops_processor.func.cfg.sl,
                       cfg.trailing_stop_rate)
    # x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to("cuda")
    x = torch.tensor(np.zeros((7, 64))).unsqueeze(0).unsqueeze(0).double().to("cuda")
    m = expert.model.double()
    m.eval()
    with torch.no_grad():
        for _ in range(10):
            torch.random.seed = 0
            print(m(x).sum())
    # print(summary(m, (10, 1, 7, 32)))