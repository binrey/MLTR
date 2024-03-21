from configs.library import *


classifier = body_classifiers.tunnel
classifier.params.ncross = Param(None, [10, 15, 20, 25, 30, 40])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])

config = EasyDict(
    lot=Param(None, [None]),
    date_start=Param("2017-08-01", ["2017-08-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(None, [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M1", ["M15"]),
    ticker=Param(None, ["BTCUSDT", "ETHUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
