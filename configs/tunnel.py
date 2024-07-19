from configs.library import *


classifier = body_classifiers.tunnel
classifier.params.ncross = Param(None, [2, 3, 4, 5, 7, 9])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(3, [True])
# stops_processor.params.tp = Param(3, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-08-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(None, [0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    allow_overturn=Param(False, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(4, [32]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param(None, ["BTCUSDT", "ETHUSDT"]),
    ticksize=Param(None, [None]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
