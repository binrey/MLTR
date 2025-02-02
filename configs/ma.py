from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.ma import ClsMACross
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop

config = dict(
    wallet=100,
    leverage=3,
    date_start="2018-01-01T00:00:00",
    date_end="2025-01-01",
    no_trading_days=set(),
    decision_maker=dict(
        type=ClsMACross,
        mode = "contrtrend",
        ma_fast_period=16,
        ma_slow_period=256,
        levels_count=8,
        levels_step=0.25
    ),
    sl_processor=dict(
        type=SLFixed,
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
        trailing_stop_rate=0.0,
    ),
    allow_overturn=False,
    hist_buffer_size=512,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=1024,
    visualize=True,
    eval_buyhold=True,
    run_model_device=None,
)


optimization = update_config(
    config, 
    symbol=[Symbols.BTCUSDT, Symbols.ETHUSDT],
    decision_maker={
        "ma_fast_period": [256, 128, 64],
        "ma_slow_period": [512, 256, 128],
        "levels_step": [0.25, 0.5, 1]
        }
    )
