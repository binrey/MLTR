from configs.library import *

classifier = body_classifiers.trend
classifier.params.npairs = Param(2, [2])
classifier.params.period = Param(4, [4, 8, 16])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-08-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    rate=Param(0.004, [0.002, 0.004, 0.008]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param("ETHUSDT", ["ETHUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
