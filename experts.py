from abc import ABC, abstractmethod
from time import perf_counter
from copy import deepcopy
import numpy as np
import yaml
from easydict import EasyDict
from loguru import logger

from indicators import ZigZag, ZigZag2, ZigZagOpt
from utils import Order
from dataloading import build_features
import torch
from utils import Broker


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
            if h.Open[-1] >= self.lprice and h.Close[-2] > self.lprice:
                self.order_dir = 1
            if self.cprice and h.Open[-1] < self.cprice:
                self.reset_state()
                return
        if self.sprice:
            if h.Open[-1] <= self.sprice and h.Close[-2] < self.sprice:
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
            print(ex)
        
        
class ClsTrend(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTrend, self).__init__(cfg, name="trend")
        self.zigzag = ZigZagOpt(max_drop=self.cfg.maxdrop)
        #self.zigzag = ZigZag2()
        
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
            # if trend_type<0:
            #     return False
            i = self.cfg.npairs*2 + 1
            common.lines = [[(x, y) for x, y in zip(ids[-i:-1], values[-i:-1])]]
            # self.get_trend(h[:-self.body_length+2])
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


class ClsTriangleSimp(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTriangleSimp, self).__init__(cfg, name="trngl_simp")
#        self.zigzag = ZigZagOpt(max_drop=0.1)
        self.zigzag = ZigZag2()
        
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
            common.lprice = max(common.lines[0][-1][1], common.lines[0][-2][1])
            common.sprice = min(common.lines[0][-1][1], common.lines[0][-2][1]) 
        return is_fig


class ClsDummy(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsDummy, self).__init__(cfg, name="dummy")
        
    def __call__(self, common, h) -> bool:
        if h.Low[-2] < h.Open[-1] < h.High[-2]:
            common.lprice = h.Close[-1] #max(h.High[-2], h.Low[-2])
            common.sprice = h.Close[-1] #min(h.High[-2], h.Low[-2])
            common.sl = {1: common.sprice, -1: common.lprice}  
            common.lines = [[(h.Id[-5], common.lprice), (h.Id[-1], common.lprice)], [(h.Id[-5], common.sprice), (h.Id[-1], common.sprice)]]
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

        
class PyConfig():
    def test(self):
        from configs.default import config
        cfg = deepcopy(config)
        for k, v in cfg.items():
            v = v.test
            if type(v) is EasyDict and "func" in v.keys():
                params = EasyDict({pk: pv.test for pk, pv in v.params.items()})
                cfg[k].func = v.func(params)
            else:
                cfg[k] = v
        return cfg

    def optim(self):
        from configs.default import config
        cfg = deepcopy(config)
        for k, vlist in cfg.items():
            vlist_new = []
            for v in vlist.optim:
                if type(v) is EasyDict and "func" in v.keys():
                    v.params = {pk: pv.optim for pk, pv in v.params.items()}
                    # v.func = partial(v.func, cfg=params)
                    params_list = self.unroll_params(v.params)
                    vlist_new += [EasyDict(func=v.func(params)) for params in params_list]
                else:
                    vlist_new.append(v)
            cfg[k] = vlist_new
        return cfg
    
    @staticmethod
    def unroll_params(cfg):
        import itertools
        keys, values = zip(*cfg.items())
        return [EasyDict(zip(keys, v)) for v in itertools.product(*values)]


class Config(EasyDict):
    def __str__(self):
        out = "config file:\n"
        for k, v in self.__dict__.items():
            if type(v) is Config:
                out += f"{k}:\n"
                for kk, vv in v.items():
                    out += f"  {kk}: {vv}\n"
            else:
                out += f"{k}: {v}\n"
        return out
    
    
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