from configs.library import *


classifier = body_classifiers.zigzag
classifier.params.feature_size = Param(16, [3, 5, 8, 12])
classifier.params.period = Param(16, [3, 5, 8])

# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(2, [2, 3, 4])
# stops_processor.params.tp = Param(10, [1, 2, 3])

stops_processor = stops_processors.stops_dynamic
# stops_processor.params.sl_active = Param(True, [True])
# stops_processor.params.tp_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2000-01-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-08-01T00:00:00", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.003, [0.002, 0.004, 0.006]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    allow_overturn=Param(True, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [128]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),

    period=Param("M60", ["M60"]),
    ticker=Param("SBER", ["BTCUSDT", "ETHUSDT"]),
    ticksize=Param(0.001, [0.001]),
    data_type=Param("metatrader", ["bybit"]),

    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
