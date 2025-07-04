import numpy as np

from common.type import (
    ConfigType,
    Symbols,
    TimePeriod,
    Vis,
    VolEstimRule,
    VolumeControl,
)
from common.utils import FeeRate, update_config
from experts.core.position_control import *
from experts.random import Random

config = dict(
    decision_maker=dict(
        type=Random,
        seed=19,
        time_to_wait=5
    ),
    sl_processor=dict(
        type=SLFixed,
        active=True,
        percent_value=0.3
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=0
    ),
    trailing_stop=dict(
        type=FixRate,
        rate=0.04,
    ),

    name="random",
    conftype=None,
    wallet=1000,
    volume_control=dict(
        type=VolumeControl,
        rule=VolEstimRule.DEPOSIT_BASED,
        deposit_fraction=0.9,
    ),
    leverage=1,
    close_only_by_stops=False,
    hist_size=32,
    traid_stops_min_size=0.1,
    tstart=0,
    tend=None,
    period=TimePeriod.M1,
    symbol=Symbols.BTCUSDT,
    data_type="bybit",
    fee_rate=FeeRate(0.055, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=128,
    visualize=True,
    run_model_device=None,
    no_trading_days=set(),
    close_last_position=True,
    handle_trade_errors=False
)

backtest = update_config(
    config,
    conftype=ConfigType.BACKTEST,
    date_start=np.datetime64("2025-06-20T05:30:00"),
    date_end=np.datetime64("2025-06-24T00:00:00"),
    eval_buyhold=False,
    clear_logs=True,
    log_trades=False,
)

optimization = update_config(
    config,
    conftype=ConfigType.OPTIMIZE,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-05-01T00:00:00"),
    clear_logs=False,
    log_trades=False,
    eval_buyhold=False,
    hist_size=[32, 64, 128, 256],
    trailing_stop={
        "rate": [0.03, 0.02, 0.01]
        },
    decision_maker={
        "sharpness": [3, 4, 5, 6],
        }
    )

bybit = update_config(
    config,
    leverage=2,
    conftype=ConfigType.BYBIT,
    credentials="bybit_volprof",
    clear_logs=False,
    log_trades=True,
    save_backup=True,
    handle_trade_errors=True,
)
