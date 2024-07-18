from configs.library import *


classifier = body_classifiers.tunzigzag
classifier.params.ncross = Param(5, [1])
classifier.params.period = Param(5, [5, 10, 15])

# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(3, [2, 3, 4])
# stops_processor.params.tp = Param(2, [1, 2, 3])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
# stops_processor.params.tp_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-08-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.0, [0.002, 0.004, 0.008]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    allow_overturn=Param(True, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    
    period=Param("M60", ["M60"]),
    ticker=Param("BTCUSDT", ["BTCUSDT"]),
    ticksize=Param(0.001, [0.001]),
    data_type=Param("bybit", ["bybit"]),
    
    save_plots=Param(True, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
