from configs.library import *


classifier = body_classifiers.levels
classifier.params.ma = Param(64, [64])
classifier.params.n = Param(5, [1, 5, 10])
classifier.params.ncross = Param(1, [1, 5, 10])
classifier.params.show_n_peaks = Param(1, [1])
classifier.params.n_extrems = Param(3, [3])

# stops_processor = stops_processors.stops_dynamic
# stops_processor.params.sl_active = Param(True, [True])

stops_processor = stops_processors.stops_fixed
stops_processor.params.sl = Param(2, [1, 2])
stops_processor.params.tp = Param(4, [1, 2])

config = EasyDict(
    lot=Param(None, [None]),
    date_start=Param("2017-10-01T00:00:00", ["2017-10-01T00:00:00"]),
    date_end=Param("2024-04-01", ["2024-04-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.00, [0.00]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(256, [256]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
