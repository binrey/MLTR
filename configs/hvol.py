from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.hvol import HVOL
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop

config = dict(
    wallet=100,
    leverage=1,
    date_start="2017-09-01T00:00:00",
    date_end="2025-01-01",
    no_trading_days=set(),
    decision_maker=dict(
        type=HVOL,
        nbins=13,
        sharpness=5
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=True,
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
    allow_overturn=False,
    hist_buffer_size=64,
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
    visualize=True,
    eval_buyhold=True,
    run_model_device=None,
)


optimization = update_config(
    config, 
    symbol=[Symbols.BTCUSDT, Symbols.ETHUSDT],
    hist_buffer_size=[32, 64, 128], 
    decision_maker={
        "nbins": [9, 11, 13, 15],
        "sharpness": [3, 4, 5]
        }
    )
