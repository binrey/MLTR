import numpy as np

from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop
from experts.vvol import VVol

config = dict(
    wallet=100,
    leverage=4,
    date_start=np.datetime64("2018-01-01T00:00:00"),
    date_end=np.datetime64("2025-03-01T00:00:00"),
    no_trading_days=set(),
    decision_maker=dict(
        type=VVol,
        nbins=9,
        sharpness=2,
        strike=4,
        strategy=VVol.TriggerStrategy.MANUAL_LEVELS
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=True,
        percent_value=0
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=2
    ),
    trailing_stop=dict(
        type=TrailingStop,
        strategy=TrailingStop.FIX_RATE,
        trailing_stop_rate=0.04,
    ),
    close_only_by_stops=True,
    hist_buffer_size=64,
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
    eval_buyhold=True,
    run_model_device=None,
)


optimization = update_config(
    config,
    min_deals_per_month=1,
    hist_buffer_size=[32, 64, 128],
    trailing_stop={
        "trailing_stop_rate": [0.04, 0.02, 0.01]
        },
    decision_maker={
        "nbins": [9, 11, 15],
        "sharpness": [2, 3, 4],
        "strike": [2, 3, 4]
        }
    )
