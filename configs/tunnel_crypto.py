from configs.library import *


config = EasyDict(
    lot=Param(0.01, [0.01]),
    date_start=Param("2017-08-01", ["2017-08-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.007, [0.004, 0.006, 0.008, 0.010, 0.012]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(body_classifiers["tunnel"], [body_classifiers[k] for k in ["tunnel"]]),
    stops_processor=Param(stops_processors["stops_dynamic"], [stops_processors[k] for k in ["stops_dynamic"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT", "ETHUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
