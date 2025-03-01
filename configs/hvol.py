from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.hvol import HVOL
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop


config = dict(
    wallet=100,
    leverage=1,
    date_start="2018-01-01T00:00:00",
    date_end="2025-01-01T00:00:00",
    no_trading_days=set(),
    decision_maker=dict(
        type=HVOL,
        nbins=20,
        sharpness=4,
        long_bin=1,
        short_bin=2,
        strategy=HVOL.TriggerStrategy.MANUAL_LEVELS
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=False,
        percent_value=2
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=2
    ),
    trailing_stop=dict(
        type=TrailingStop,
        strategy=TrailingStop.FIX_RATE,
        trailing_stop_rate=0.01,
    ),
    close_only_by_stops=False,
    hist_buffer_size=128,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT,
    equaty_step=0.001,
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
    symbol=[Symbols.BTCUSDT],
    hist_buffer_size=[32, 64, 128], 
    decision_maker={
        "nbins": [10, 20, 30],
        "sharpness": [3, 4, 5],
        "long_bin": [1, 2, 3],
        "short_bin": [1, 2, 3]
        }
    )
