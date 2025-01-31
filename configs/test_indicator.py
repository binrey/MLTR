from common.type import Symbols, TimePeriod, Vis
from common.utils import FeeRate, update_config
from experts.hvol import HVOL
from experts.position_control import SLDynamic, SLFixed, TPFromSL, TrailingStop

config = dict(
    wallet=50,
    leverage=1,
    date_start="2024-01-01T00:00:00",
    date_end="2024-11-01",
    no_trading_days=set(),
    trailing_stop_rate=0.004,
    decision_maker=dict(
        type=HVOL,
        ncross=3
    ),
    allow_overturn=False,
    sl_processor=dict(
        type=SLDynamic,
        active=True
    ),
    hist_buffer_size=64,
    tstart=0,
    tend=None,
    period=TimePeriod.M60,
    symbol=Symbols.BTCUSDT,
    equaty_step=0.01,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=256,
    visualize=True,
    eval_buyhold=False,
    run_model_device=None,
)

optimization = config.copy()
optimization.update(dict(
    hist_buffer_size=[32, 64]
))