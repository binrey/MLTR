from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop
from experts.vvol_plus import VVolPlus

config = dict(
    wallet=100,
    leverage=1,
    date_start="2018-01-01T00:00:00",
    date_end="2025-01-01T00:00:00",
    no_trading_days=set(),
    decision_maker=dict(
        type=VVolPlus,
        nbins=20,
        sharpness=4,
        long_bin=0,
        short_bin=0,
        strategy=VVolPlus.TriggerStrategy.MANUAL_LEVELS,
        zigzag_period=4
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
        rate=0.001,
    ),
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
    eval_buyhold=True,
    run_model_device=None,
)


optimization = update_config(
    config,
    symbol=[Symbols.BTCUSDT],
    hist_size=[128],
    decision_maker={
        "nbins": [20],
        "sharpness": [3, 4, 5, 7],
        "zigzag_period": [3, 5, 8],
        }
    )
