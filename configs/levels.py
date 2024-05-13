from configs.library import *


classifier = body_classifiers.levels
classifier.params.ma = Param(64, [64])
classifier.params.n = Param(5, [0, 5, 15])
classifier.params.ncross = Param(0, [0, 5, 15])
classifier.params.show_n_peaks = Param(3, [4])

# stops_processor = stops_processors.stops_dynamic
# stops_processor.params.sl_active = Param(True, [True])

stops_processor = stops_processors.stops_fixed
stops_processor.params.sl = Param(3, [2, 3, 4])
# stops_processor.params.tp = Param(4, [True])

config = EasyDict(
    lot=Param(None, [None]),
    date_start=Param("2017-09-29T00:00:00", ["2017-09-01T00:00:00"]),
    date_end=Param("2024-06-01", ["2024-05-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.004, [0.002, 0.004, 0.008]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(8, [8]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT", "ETHUSDT"]),
    data_type=Param("bybit", ["metatrader"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
