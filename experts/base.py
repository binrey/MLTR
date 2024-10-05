from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from loguru import logger

# import torch
from backtesting.backtest_broker import Broker, Order, Position
from common.type import Side
from common.utils import date2str
from indicators import *
from trade.utils import ORDER_TYPE, fix_rate_trailing_sl


def log_modify_sl(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, sl: Optional[float]):
        logger.info(f"Modifying sl: {self.active_position.sl} -> {sl}")
        result = func(self, sl)
        return result
    return wrapper

class ExpertBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.body_cls = cfg.body_classifier.func
        self.sl_processor = cfg.sl_processor.func
        self.sl_processor.set_expert(self)
        
        self.lines = []
        self.orders = []
            
    @abstractmethod
    def get_body(self) -> None:
        pass
    
    @abstractmethod
    def create_orders(self) -> None:
        pass
    
    def update(self, h, active_position: Position):
        self.active_position = active_position
        self.get_body(h)


class ExtensionBase:
    def __init__(self, cfg, name):
         self.name = name + ":" + "-".join([f"{v}" for k, v in cfg.items()])
         
    def __call__(self, common, h):
        pass
            
    def update_inner_state(self, h):
        pass


class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        super(ExpertFormation, self).__init__(cfg)  
        self._reset_state()
        self.order_sent = False
        self.sl = None
        self.tp = None
        
        if self.cfg.run_model_device is not None:
            from ml import Net, Net2
            self.model = Net2(4, 32)
            self.model.load_state_dict(torch.load("model.pth"))
            # self.model.set_threshold(0.6)
            self.model.eval()
            self.model.to(self.cfg.run_model_device)
        
    def _reset_state(self):
        self.lprice = None
        self.sprice = None
        self.cprice = None
        self.formation_found = False
        self.order_dir = 0
            
    def estimate_volume(self, h):
        volume = self.cfg.wallet/h["Open"][-1]*self.cfg.leverage
        volume = self.normalize_volume(volume)
        logger.info(f"estimated lot: {volume}")
        return volume
    
    def normalize_volume(self, volume):
        return round(volume/self.cfg.ticksize, 0)*self.cfg.ticksize
            
    def update_trailing_sl(self, h):
        if self.active_position is not None:
            if self.active_position.sl is None:
                sl = self.sl_processor(h)
                self.modify_sl(sl)
            else:
                sl_new = fix_rate_trailing_sl(sl=self.active_position.sl, 
                                              open_price=h["Open"][-1],
                                              side=self.active_position.side,
                                              trailing_stop_rate=self.cfg.trailing_stop_rate, 
                                              ticksize=self.cfg.ticksize)
                self.modify_sl(sl_new)                

    def get_body(self, h):
        self.update_trailing_sl(h)
        self.body_cls.update_inner_state(h)
        if not self.cfg.allow_overturn and self.active_position is not None:
            return
        
        self.order_sent = False
        self.order_dir = 0
        
        if self.cfg.allow_overturn or not self.formation_found:
            self.formation_found = self.body_cls(self, h)   
        
        logger.info(f"found enter points: long: {self.lprice}, short: {self.sprice}, cancel: {self.cprice}")
        
        if self.lprice:
            if (self.sprice is None and h["Open"][-1] >= self.lprice) or h["Close"][-2] > self.lprice:
                self.order_dir = 1
            if self.cprice is not None and h["Open"][-1] < self.cprice:
                self._reset_state()
                return
            
        if self.sprice:
            if (self.lprice is None and h["Open"][-1] <= self.sprice) or h["Close"][-2] < self.sprice:
                self.order_dir = -1
            if self.cprice and h["Open"][-1] > self.cprice:
                self._reset_state()
                return            
        
        if h["Date"][-1] in self.cfg.no_trading_days:
            self._reset_state()
            
        # y = None
        # if self.cfg.run_model_device and self.order_dir != 0:
        #     x = build_features(h, 
        #                        self.order_dir, 
        #                        self.stops_processor.cfg.sl,
        #                        self.cfg.trailing_stop_rate
        #                        )
        #     x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(self.cfg.run_model_device)
        #     y = [0.5, 1, 2, 4, 8][self.model.predict(x).item()]
            
        
        if self.order_dir != 0:
            if self.active_position is None or self.active_position.side.value*self.order_dir < 0:
                
                self.create_orders(side=Side.from_int(self.order_dir), 
                                   volume=self.estimate_volume(h),
                                   time_id=h["Id"][-1])
                self.order_sent = True
                if not self.cfg.allow_overturn:
                    self._reset_state()
            

            
class BacktestExpert(ExpertFormation):
    def __init__(self, cfg, session: Broker):
        self.cfg = cfg
        self.session = session
        super(BacktestExpert, self).__init__(cfg)
        
    def create_orders(self, *, side, volume, time_id):
        self.orders = [Order(0, side, ORDER_TYPE.MARKET, volume, time_id, time_id)]
        log_message = f"{time_id} send order {self.orders[0]}"

        self.session.set_active_orders(self.orders)
        logger.info(log_message)
    
    @log_modify_sl    
    def modify_sl(self, sl: Optional[float]):
        self.session.update_sl(sl)
        
            
class ByBitExpert(ExpertFormation):
    def __init__(self, cfg, session):
        self.cfg = cfg
        self.session = session
        super(ByBitExpert, self).__init__(cfg)
        
    def create_orders(self, *, side, volume, time_id=None):
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=self.cfg.ticker,
                side=side.name.capitalize(),
                orderType="Market",
                qty=volume,
                timeInForce="GTC",
                # orderLinkId="spot-test-po1stonly",
                # stopLoss="" if sl is None else str(abs(sl)),
                # takeProfit="" if tp is None else str(tp)
                )
            logger.info(f"place order result: {resp.get('result', '--')}")
        except Exception as ex:
            logger.error(ex)

    @log_modify_sl 
    def modify_sl(self, sl: Optional[float]):
        if sl is None:
            return
        try:
            resp = self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg.ticker,
                stopLoss=sl,
                slTriggerB="IndexPrice",
                positionIdx=0,
            )
        except Exception as ex:
            logger.error(ex)


