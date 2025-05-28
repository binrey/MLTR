import numpy as np

from common.type import ConfigType, Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.buy_and_hold import BuyAndHold
from experts.core.position_control import *

config = dict(
    decision_maker=dict(
        type=BuyAndHold,
    ),
    sl_processor=dict(
        type=SLFixed,
        active=False,
        percent_value=0
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=0
    ),
    trailing_stop=dict(
        type=FixRate,
        rate=0,
    ),
    
    name="buy_and_hold",
    conftype=None,
    wallet=1000,
    leverage=1,
    close_only_by_stops=False,
    hist_size=2,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=1,
    visualize=False,
    run_model_device=None,
    no_trading_days=set(),
    handle_trade_errors=False,
)

backtest = update_config(
    config,
    conftype=ConfigType.BACKTEST,
    date_start=np.datetime64("2020-01-01T00:00:00"),
    date_end=np.datetime64("2025-05-21T00:00:00"),
    eval_buyhold=True,
    clear_logs=True,
    log_trades=False,
    close_last_position=True,
)
