from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.ma_cross import ClsMACross
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop

config = dict(
    wallet=100,
    lot=0.02,
    leverage=1,
    date_start="2018-01-01T00:00:00",
    date_end="2025-01-01",
    no_trading_days=set(),
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
        rate=0.0,
    ),
    close_only_by_stops=False,
    hist_size=1028,
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
    visualize=True,
    eval_buyhold=True,
    run_model_device=None,
)


optimization = update_config(
    config, 
    symbol=[Symbols.BTCUSDT, Symbols.ETHUSDT],
    decision_maker={
        "ma_fast_period": [512, 256, 128, 64],
        "levels_count": [3, 5, 10, 20, 30],
        "levels_step": [1, 1.5, 2, 3]
        }
    )
