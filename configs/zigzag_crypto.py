from configs.library import *


classifier = body_classifiers.trend
classifier.params.npairs = Param(2, [2, 3])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(2, [1]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-07-01", ["2024-07-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.004, [0.002, 0.003, 0.004, 0.005, 0.006]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param("BTCUSDT", ["BTCUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.1, [0.1]),
    eval_buyhold=Param(True, [True]),
    fuse_buyhold=Param(True, [True]),
)
