from configs.library import *
from utils import FeeRate

classifier = body_classifiers.tunnel
classifier.params.ncross = Param(4, [3, 4, 5, 7, 9])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(3, [True])
# stops_processor.params.tp = Param(3, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(10, [1]),
    date_start=Param("2000-01-01T00:00:00", ["2000-01-01"]),
    date_end=Param("2024-10-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.006, [0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    allow_overturn=Param(False, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    hist_buffer_size=Param(64, [32, 64, 128]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param("BTCUSDT", ["ETHUSDT", "BTCUSDT"]),
    ticksize=Param(0.001, [0.001]),
    data_type=Param("metatrader", ["metatrader"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(FeeRate(0.1, 0.00016), [FeeRate(0.1, 0.00016)]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
