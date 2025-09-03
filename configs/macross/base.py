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
from experts.ma_cross import ClsMACross

config = dict(
    decision_maker=dict(
        type=ClsMACross,
        mode = "contrtrend",
        ma_fast_period=16,
        upper_levels = 3,
        lower_levels = 10,
        min_step=0.25,
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=False,
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
    ),
    trailing_stop=dict(
        type=FixRate,
        rate=0.05,
    ),
    
    name="macross",
    conftype=None,
    wallet=1000,
    lot=0.08,
    leverage=1,
    volume_control=dict(
        type=VolumeControl,
        rule=VolEstimRule.FIXED_POS_COST,
        deposit_fraction=0.33,
    ),
    close_only_by_stops=False,
    hist_size=256,
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
    date_end=np.datetime64("2025-06-01T00:00:00"),
    eval_buyhold=True,
    clear_logs=True,
    log_trades=True,
    verify_data=False,
)

optimization = update_config(
    config,
    conftype=ConfigType.OPTIMIZE,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-05-01T00:00:00"),
    clear_logs=False,
    log_trades=False,
    eval_buyhold=False,
    hist_size=[256],
    lot=[0.02, 0.04, 0.08],
    decision_maker={
        "ma_fast_period": [16, 32, 64],
        "upper_levels": [0, 5, 10],
        "lower_levels": [20, 30, 40],
        "min_step": [0.125, 0.25, 0.5],
        # "speed": [0.125, 0.25, 0.5],
        }
    )

bybit = update_config(
    config,
    conftype=ConfigType.BYBIT,
    credentials="bybit_volprof",
    clear_logs=False,
    log_trades=True,
    save_backup=True,
    handle_trade_errors=True,
)
