from configs.library import *

classifier = body_classifiers.levels
classifier.params.ma = Param(16, [64])
classifier.params.n = Param(1, [1, 5, 10])
classifier.params.ncross = Param(1, [1, 5, 10])
classifier.params.show_n_peaks = Param(6, [1])
classifier.params.n_extrems = Param(16, [3])

# stops_processor = stops_processors.stops_dynamic
# stops_processor.params.sl_active = Param(True, [True])

stops_processor = stops_processors.stops_fixed
stops_processor.params.sl = Param(2, [1, 2])
stops_processor.params.tp = Param(4, [1, 2])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-08-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    rate=Param(0.003, [0.001, 0.002, 0.004]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param("ETHUSDT", ["ETHUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(True, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
