from typing import Any, Callable, Optional

from loguru import logger

# import torch
from backtesting.backtest_broker import Broker, Order
from common.type import Side
from experts.core.expert import ExpertBase
from indicators import *
from trade.utils import ORDER_TYPE


def log_modify_sl(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, sl: Optional[float]):
        logger.debug(f"Modifying sl: {self.active_position.sl} -> {sl}")
        result = func(self, sl)
        return result
    return wrapper

class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        super(ExpertFormation, self).__init__(cfg)  
        self._reset_state()
        self.order_sent = False
        
        if self.cfg["run_model_device"] is not None:
            from ml import Net, Net2
            self.model = Net2(4, 32)
            self.model.load_state_dict(torch.load("model.pth"))
            # self.model.set_threshold(0.6)
            self.model.eval()
            self.model.to(self.cfg["run_model_device"])
        
    def _reset_state(self):
        self.lprice = None
        self.sprice = None
        self.cprice = None
        self.formation_found = False
        self.order_dir = 0
            
    def estimate_volume(self, h):
        volume = self.cfg["wallet"]/h["Open"][-1]*self.cfg["leverage"]
        volume = self.normalize_volume(volume)
        logger.debug(f"estimated lot: {volume}")
        return volume
    
    def normalize_volume(self, volume):
        return round(volume/self.cfg["symbol"].qty_step, 0)*self.cfg["symbol"].qty_step
            
    def create_or_update_sl(self, h):
        if self.active_position is not None:
            if self.active_position.sl is None:
                sl = self.sl_processor.create(hist=h,
                                              active_position=self.active_position,
                                              decision_maker=self.decision_maker)
                self.modify_sl(sl)
            else:
                sl_new = self.trailing_stop.get_stop_loss(self.active_position, hist=h)
                self.modify_sl(sl_new)                

    def create_or_update_tp(self, h):
        if self.active_position is not None:
            if self.active_position.tp is None:
                tp = self.tp_processor.create(hist=h,
                                              active_position=self.active_position,
                                              decision_maker=self.decision_maker)
                self.modify_tp(tp)

    def get_body(self, h):
        self.create_or_update_sl(h)
        self.create_or_update_tp(h)
        self.decision_maker.update_inner_state(h)
        if not self.cfg["allow_overturn"] and self.active_position is not None:
            return
        
        self.order_sent = False
        self.order_dir = 0
        
        if self.cfg["allow_overturn"] or not self.formation_found:
            if self.decision_maker.look_around(h):
                self.formation_found = True
                self.lprice = self.decision_maker.lprice
                self.sprice = self.decision_maker.sprice
        
        logger.debug(f"found enter points: long: {self.lprice}, short: {self.sprice}, cancel: {self.cprice}")
        
        if self.lprice:
            if (self.sprice is None and h["Open"][-1] >= self.lprice) or h["Close"][-2] > self.lprice:
                if h["Close"][-3] < self.lprice:
                    self.order_dir = 1
            if self.cprice is not None and h["Open"][-1] < self.cprice:
                self._reset_state()
                return
            
        if self.sprice:
            if (self.lprice is None and h["Open"][-1] <= self.sprice) or h["Close"][-2] < self.sprice:
                if h["Close"][-3] > self.sprice:
                    self.order_dir = -1
            if self.cprice and h["Open"][-1] > self.cprice:
                self._reset_state()
                return            
        
        if h["Date"][-1] in self.cfg["no_trading_days"]:
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
                if not self.cfg["allow_overturn"]:
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
        logger.debug(log_message)
    
    @log_modify_sl    
    def modify_sl(self, sl: Optional[float]):
        self.session.update_sl(sl)

    def modify_tp(self, tp: Optional[float]):
        self.session.update_tp(tp)        
            
class ByBitExpert(ExpertFormation):
    def __init__(self, cfg, session):
        self.cfg = cfg
        self.session = session
        super(ByBitExpert, self).__init__(cfg)
        
    def create_orders(self, *, side, volume, time_id=None):
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=self.cfg["symbol"].ticker,
                side=side.name.capitalize(),
                orderType="Market",
                qty=volume,
                timeInForce="GTC",
                # orderLinkId="spot-test-po1stonly",
                # stopLoss="" if sl is None else str(abs(sl)),
                # takeProfit="" if tp is None else str(tp)
                )
            logger.debug(f"place order result: {resp.get('result', '--')}")
        except Exception as ex:
            logger.error(ex)

    @log_modify_sl 
    def modify_sl(self, sl: Optional[float]):
        if sl is None:
            return
        try:
            resp = self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg["symbol"].ticker,
                stopLoss=sl,
                slTriggerB="IndexPrice",
                positionIdx=0,
            )
        except Exception as ex:
            logger.error(ex)