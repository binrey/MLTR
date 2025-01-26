from copy import deepcopy

from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop, TrailingStopStrategy
from experts.tunnel import ClsTunnel

config = dict(
    wallet=100,
    leverage=3,
    date_start="2017-09-01T00:00:00",
    date_end="2025-01-01",
    allow_overturn=False,
    
    decision_maker=dict(
        type=ClsTunnel,
        ncross=4
    ),
    sl_processor=dict(
        type=SLDynamic,
        active=True
    ),
    tp_processor=dict(
        type=TPFromSL,
        active=False,
        scale=0
    ),
    trailing_stop=dict(
        type=TrailingStop,
        strategy=TrailingStop.FIX_RATE,
        trailing_stop_rate=0.002,
    ),
    hist_buffer_size=32,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT.value,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=256,
    visualize=False,
    eval_buyhold=True,
    
    run_model_device=None,
    no_trading_days=set(),
)

optimization = deepcopy(config)
optimization["hist_buffer_size"] = [64]
optimization["trailing_stop_rate"] = [0.002]
optimization["decision_maker"]["ncross"] = [4, 5, 7, 9]
optimization["symbol"] = [Symbols.BTCUSDT.value, 
                          Symbols.ETHUSDT.value]