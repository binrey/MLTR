from typing import Any, Callable, Optional

from loguru import logger
from pybit.unified_trading import HTTP

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

def log_modify_tp(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self, tp: Optional[float]):
        logger.debug(f"Modifying tp: {self.active_position.tp} -> {tp}")
        result = func(self, tp)
        return result
    return wrapper

class ExpertFormation(ExpertBase):
    def __init__(self, cfg):
        super(ExpertFormation, self).__init__(cfg)  
        self.traid_stops_min_size_multiplier = 3
        # if self.cfg["run_model_device"] is not None:
        #     from ml import Net, Net2
        #     self.model = Net2(4, 32)
        #     self.model.load_state_dict(torch.load("model.pth"))
        #     # self.model.set_threshold(0.6)
        #     self.model.eval()
        #     self.model.to(self.cfg["run_model_device"])
            
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
            else:
                sl = self.trailing_stop.get_stop_loss(self.active_position, hist=h)
            
            if self.active_position.side == Side.BUY:
                sl = min(sl, h["Open"][-1] - self.symbol.tick_size*self.traid_stops_min_size_multiplier)
            else:
                sl = max(sl, h["Open"][-1] + self.symbol.tick_size*self.traid_stops_min_size_multiplier)
            self.modify_sl(sl)                

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
        
        if self.cfg["close_only_by_stops"] and self.active_position:
            return

        target_state = self.decision_maker.look_around(h)
        
        if h["Date"][-1] in self.cfg["no_trading_days"]:
            self._reset_state()
            
        # y = None
        # if self.cfg.run_model_device and self.order_dir != 0:
        #     x = build_features(h, 
        #                        self.order_dir, 
        #                        self.stops_processor.cfg.sl,
        #                        self.cfg.rate
        #                        )
        #     x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(self.cfg.run_model_device)
        #     y = [0.5, 1, 2, 4, 8][self.model.predict(x).item()]
            
        
        if not target_state.is_active:
            return

        max_volume = self.estimate_volume(h)  
                   
        if target_state.target_volume_fraction is not None:
            target_volume = target_state.target_volume_fraction
            if self.active_position:
                side_relative = target_state.side.value * self.active_position.side.value
                target_volume *= side_relative
        else:
            if target_state.increment_volume_fraction is not None:
                addition = target_state.increment_volume_fraction
            else:
                addition = target_state.increment_by_num_lots * self.cfg["lot"]   
                
            target_volume = addition             
            if self.active_position:
                side_relative = target_state.side.value * self.active_position.side.value
                if side_relative > 0:
                    target_volume = min(1, self.active_position.volume / max_volume + addition)
                if side_relative < 0:
                    target_volume = max(-1, self.active_position.volume / max_volume - addition)
                    # Do not trim position:
                    target_volume = -addition
        
        target_volume *= max_volume

        if self.active_position is None:
            # Open new position
            order_volume = target_volume
        else:
            if side_relative < 0:
                # Close old and open new
                order_volume = self.active_position.volume - target_volume
                # if order_volume < self.active_position.volume:
                #     return
            else:
                # Add to old
                order_volume = max(0, target_volume - self.active_position.volume)
        if order_volume > 0:
            self.create_orders(side=target_state.side, 
                               volume=order_volume,
                               time_id=h["Id"][-1])
            
            
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

    @log_modify_tp
    def modify_tp(self, tp: Optional[float]):
        self.session.update_tp(tp)        
            
class ByBitExpert(ExpertFormation):
    def __init__(self, cfg, session: HTTP):
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
            self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg["symbol"].ticker,
                stopLoss=sl
            )
        except Exception as ex:
            logger.error(ex)
            
    @log_modify_tp
    def modify_tp(self, tp: Optional[float]):
        if tp is None:
            return
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg["symbol"].ticker,
                takeProfit=tp
            )
        except Exception as ex:
            logger.error(ex)