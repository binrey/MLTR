from configs.library import *


classifier = body_classifiers.tunnel
classifier.params.ncross = Param(None, [3, 5, 7, 10])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    date_start=Param("2017-09-01T00:00:00", ["2017-09-01"]),
    date_end=Param("2024-07-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(None, [0.002, 0.003, 0.004, 0.005, 0.006]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M60", ["M60"]),
    ticker=Param(None, ["BTCUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(0.05, [0.05]),
    eval_buyhold=Param(True, [True]),
    fuse_buyhold=Param(True, [True]),
)
