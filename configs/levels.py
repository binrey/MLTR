from configs.library import *


classifier = body_classifiers.levels
classifier.params.ma = Param(64, [64])
classifier.params.n = Param(30, [4])

stops_processor = stops_processors.stops_fixed
stops_processor.params.sl = Param(3, [True])
# stops_processor.params.tp = Param(4, [True])

config = EasyDict(
    lot=Param(None, [None]),
    date_start=Param("2017-09-01T00:00:00", ["2017-08-01"]),
    date_end=Param("2024-05-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.005, [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(256, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT", "ETHUSDT"]),
    data_type=Param("metatrader", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
