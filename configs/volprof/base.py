import numpy as np

from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.position_control import FixRate, SLDynamic, SLFixed, TPFromSL, TrailingStop
from experts.volprof import VolProf

config = dict(
    wallet=100,
    leverage=1,
    decision_maker=dict(
        type=VolProf,
        sharpness=0,
        strategy=VolProf.Levels.MANUAL
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
    close_only_by_stops=False,
    hist_size=64,
    tstart=0,
    tend=None,
    period=TimePeriod.M1,
    symbol=Symbols.BTCUSDT,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=512,
    visualize=False,
    run_model_device=None,
)

backtest = update_config(
    config,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-05-01T00:00:00"),
    no_trading_days=set(),
    eval_buyhold=True,
)

optimization = update_config(
    config,
    hist_size=[32, 64, 128, 256],
    trailing_stop={
        "rate": [0.03, 0.02, 0.01]
        },
    decision_maker={
        "sharpness": [3, 4, 5, 6],
        }
    )

trading = update_config(
    config,
    credentials="bybit_volprof",
)
