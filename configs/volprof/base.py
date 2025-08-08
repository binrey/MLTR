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
from experts.volprof import VolProf

config = dict(
    decision_maker=dict(
        type=VolProf,
        sharpness=0,
        demo=False
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=True,
        percent_value=0
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=0
    ),
    trailing_stop=dict(
        type=FixRate,
        rate=0.02,
    ),

    name="volprof",
    conftype=None,
    wallet=1000,
    volume_control=dict(
        type=VolumeControl,
        rule=VolEstimRule.FIXED_POS_COST,
        deposit_fraction=1,
    ),
    leverage=1,
    close_only_by_stops=False,
    hist_size=64,
    traid_stops_min_size=0.2,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=512,
    visualize=False,
    run_model_device=None,
    no_trading_days=set(),
    close_last_position=True,
    handle_trade_errors=False
)

backtest = update_config(
    config,
    conftype=ConfigType.BACKTEST,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-07-06T00:00:00"),
    eval_buyhold=False,
    clear_logs=True,
    log_trades=True,
)

optimization = update_config(
    config,
    conftype=ConfigType.OPTIMIZE,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-07-06T00:00:00"),
    clear_logs=False,
    log_trades=False,
    eval_buyhold=False,
    symbol=[Symbols.BTCUSDT, Symbols.ETHUSDT],
    hist_size=[64, 128, 256],
    trailing_stop={
        "rate": [0.01, 0.02, 0.04]
        },
    decision_maker={
        "sharpness": [3, 4, 5],
        }
    )

bybit = update_config(
    config,
    leverage=1,
    conftype=ConfigType.BYBIT,
    credentials="bybit_volprof",
    clear_logs=False,
    log_trades=True,
    save_backup=True,
    handle_trade_errors=True,

)
