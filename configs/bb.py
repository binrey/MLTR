from configs.library import *

classifier = body_classifiers.bb

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
stops_processor.params.tp_active = Param(True, [True])

config = EasyDict(
    lot=Param(None, [None]),
    date_start=Param("2019-08-01", ["2017-08-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    rate=Param(0., [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_size=Param(128, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT", "ETHUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
