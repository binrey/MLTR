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
    decision_maker=dict[str, type[ClsMACross] | str | int | float](
        type=ClsMACross,
        mode = "contrtrend",
        ma_fast_period=16,
        upper_levels = 0,
        lower_levels = 1,
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
        rate=0.,
    ),
    
    name="macross",
    conftype=None,
    wallet=10000,
    lot=1,
    leverage=1,
    volume_control=dict(
        type=VolumeControl,
        rule=VolEstimRule.FIXED_POS_COST,
        deposit_fraction=1.,
    ),
    close_only_by_stops=False,
    hist_size=256,
    traid_stops_min_size=0.2,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.ETHUSDT,
    data_type="bybit",
    fee_rate=FeeRate(
        order_execution_rate=0.1, 
        order_execution_slippage_rate=0.015, 
        position_suply_rate=0.00016),    
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
    date_end=np.datetime64("2025-11-01T00:00:00"),
    clear_logs=True,
    log_trades=False,
    verify_data=False,
    log_config=False,
)

optimization = update_config(
    config,
    conftype=ConfigType.OPTIMIZE,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-11-01T00:00:00"),
    clear_logs=False,
    log_trades=False,
    eval_buyhold=False,
    log_config=False,
    find_best_multistrategy=True,
    param2sort="recovery",
    lot=[0.2, 0.4, 0.6, 0.8, 1.0],
    hist_size=[200, 300, 400, 500, 600, 800],
    decision_maker={
        "ma_fast_period": [5, 10, 20, 30, 40, 60, 80, 100],
        "min_step": [0.4, 0.6, 0.8, 1],
        }
    )

bybit = update_config(
    config,
    leverage=1,
    conftype=ConfigType.BYBIT,
    clear_logs=False,
    log_trades=True,
    save_backup=True,
    handle_trade_errors=True,
    verify_data=True,
    log_config=True,
)
