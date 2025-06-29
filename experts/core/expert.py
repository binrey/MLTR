from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable

from loguru import logger

# import torch
from backtesting.backtest_broker import Position
from common.type import Side, Symbol, VolEstimRule
from experts.core.decision_maker import DecisionMaker
from experts.core.position_control import StopsController, TrailingStop

# from indicators import *


def init_target_from_cfg(cfg):
    cfg = deepcopy(cfg)
    Target = cfg.pop("type")
    return Target(cfg)


class Expert:
    def __init__(self,
                 cfg: dict[str, Any],
                 create_orders_func: Callable[[Side, float, int], None],
                 modify_sl_func: Callable[[float], None],
                 modify_tp_func: Callable[[float], None]):
        self.symbol: Symbol = cfg["symbol"]
        self.close_only_by_stops = cfg["close_only_by_stops"]
        self.no_trading_days = cfg["no_trading_days"]
        self.hist_size = cfg["hist_size"]
        self.period = cfg["period"]
        self.wallet = cfg["wallet"]
        self.leverage = cfg["leverage"]
        self.lot = cfg.get("lot", None)
        self.volume_control: VolEstimRule = cfg["volume_control"]
        
        decision_maker_cfg = cfg["decision_maker"].copy()
        decision_maker_cfg.update(
            {k: cfg[k] for k in ["hist_size", "period", "symbol"]})
        self.decision_maker: DecisionMaker = init_target_from_cfg(
            decision_maker_cfg)

        assert "sl_processor" in cfg, "sl_processor must be defined in cfg"
        self.sl_processor: StopsController = init_target_from_cfg(
            cfg["sl_processor"])
        self.sl = None

        assert "tp_processor" in cfg, "tp_processor must be defined in cfg"
        self.tp_processor: StopsController = init_target_from_cfg(
            cfg.get("tp_processor", None))
        self.tp = None

        assert "trailing_stop" in cfg, "trailing_stop must be defined in cfg"
        self.trailing_stop: TrailingStop = init_target_from_cfg(
            cfg.get("trailing_stop", None))

        self.orders = []
        self.active_position = None
        self.deposit = self.wallet
        
        self.create_orders = create_orders_func
        self.modify_sl = modify_sl_func
        self.modify_tp = modify_tp_func

    def __str__(self):
        return f"{str(self.decision_maker)} sl: {str(self.sl_processor)}  tp: {str(self.tp_processor)}"

    def update(self, h, active_position: Position, deposit: float):
        self.active_position = active_position
        self.deposit = deposit
        self.get_body(h)

    def _reset_state(self):
        """Resets the expert state on no trading days."""
        self.active_position = None
        self.orders = []

    def estimate_volume(self, h):
        if self.volume_control.rule is VolEstimRule.FIXED_POS_COST:
            base = self.volume_control.define(self.wallet)
        elif self.volume_control.rule is VolEstimRule.DEPOSIT_BASED:
            base = self.volume_control.define(self.deposit)
        else:
            raise ValueError(f"Unknown volume estimation rule: {self.volume_control.rule}")
        
        volume = base / h["Open"][-1] * self.leverage
        volume_norm = Symbol.round_qty(qty=volume, qty_step=self.symbol.qty_step)
        message = f"Estimated lot: {volume_norm} <- {volume} <- ({base:.2f}$ / price: {h['Open'][-1]} * leverage: {self.leverage}). Cost: {volume*h['Open'][-1]:.2f}"
        if volume == 0:
            logger.warning(message)
        else:
            logger.debug(message)
        return volume_norm

    def create_or_update_sl(self, h):
        if self.active_position is not None:
            if self.active_position.sl is None:
                sl = self.sl_processor.create(open_price=h["Close"][-2],
                                              active_position=self.active_position,
                                              decision_maker=self.decision_maker,
                                              tick_size=self.symbol.tick_size)
            else:
                sl = self.trailing_stop.get_stop_loss(
                    self.active_position, hist=h)

            if sl != self.active_position.sl:
                self.modify_sl(sl, h["Close"][-2])

    def create_or_update_tp(self, h):
        if self.active_position is not None:
            if self.active_position.tp is None:
                tp = self.tp_processor.create(hist=h,
                                              active_position=self.active_position,
                                              decision_maker=self.decision_maker)
                if tp != self.active_position.tp:
                    self.modify_tp(tp)

    def get_body(self, h):
        self.create_or_update_sl(h)
        self.create_or_update_tp(h)
        self.decision_maker.update_inner_state(h)

        if self.close_only_by_stops and self.active_position:
            return

        target_state: DecisionMaker.Response = self.decision_maker.look_around(h)

        if h["Date"][-1] in self.no_trading_days:
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
                if target_state.side.value == 0:
                    target_state.side = Side.reverse(self.active_position.side)
                    side_relative = target_state.side.value
                else: 
                    side_relative = target_state.side.value * self.active_position.side.value
                target_volume *= side_relative
        else:
            if target_state.increment_volume_fraction is not None:
                addition = target_state.increment_volume_fraction
            elif target_state.increment_by_num_lots is not None:
                addition = target_state.increment_by_num_lots * self.lot
            else:
                raise ValueError(target_state)

            target_volume = addition
            if self.active_position:
                side_relative = target_state.side.value * self.active_position.side.value
                if side_relative > 0:
                    target_volume = min(
                        1, self.active_position.volume / max_volume + addition)
                if side_relative < 0:
                    target_volume = max(-1, self.active_position.volume /
                                        max_volume - addition)
                    # Do not trim position:
                    target_volume = -addition

        if target_volume > 1:
            target_volume = 1
            logger.warning("try to set up target volume more than max value...set to one")
        target_volume *= max_volume

        order_volume = 0
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
                # Add to position
                if target_volume == 0:
                    order_volume = max(0, target_volume - self.active_position.volume)

        if order_volume > 0:
            self.create_orders(side=target_state.side,
                               volume=order_volume,
                               time_id=h["Id"][-1])