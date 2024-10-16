from common.type import TimePeriod, Vis
from common.utils import FeeRate
from experts.hvol import HVOL
from experts.position_control import SLDynamic

config = dict(
    wallet=100,
    leverage=1,
    date_start="2000-01-01T00:00:00",
    date_end="2024-11-01",
    no_trading_days=set(),
    trailing_stop_rate=0.0025,
    decision_maker=dict(
        type=HVOL,
        nbins=30,
        sharpness=3
    ),
    allow_overturn=False,
    sl_processor=dict(
        type=SLDynamic,
        active=True
    ),
    hist_buffer_size=128,
    tstart=0,
    tend=None,
    period=TimePeriod.M15,
    ticker="BTCUSDT",
    ticksize=0.001,
    data_type="bybit",
    fee_rate=FeeRate(0.1, 0.00016),
    save_backup=False,
    save_plots=False,
    vis_events=Vis.ON_DEAL,
    vis_hist_length=512,
    visualize=False,
    eval_buyhold=False,
    run_model_device=None,
)

optimization = config.copy()
optimization.update(dict(
    hist_buffer_size=[32, 64]
))