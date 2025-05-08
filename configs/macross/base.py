import numpy as np

from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.ma_cross import ClsMACross
from experts.position_control import FixRate, SLDynamic, SLFixed, TPFromSL, TrailingStop

config = dict(
    decision_maker=dict(
        type=ClsMACross,
        mode = "contrtrend",
        ma_fast_period=16,
        ma_slow_period=256,
        upper_levels = 2,
        lower_levels = 20,
        min_step=0.25,
        speed=0.5
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
    wallet=100,
    leverage=1,
    close_only_by_stops=False,
    hist_size=64,
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
)

backtest = update_config(
    config,
    conftype="backtest",
    date_start=np.datetime64("2025-01-03T22:28:34"),
    date_end=np.datetime64("2025-05-01T00:00:00"),
    eval_buyhold=True,
    clear_logs=True,
    log_trades=True,
    close_last_position=True,
)

optimization = update_config(
    config,
    conftype="optimization",
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
    conftype="bybit",
    credentials="bybit_volprof",
    clear_logs=False,
    log_trades=True,
    save_backup=True,
)
