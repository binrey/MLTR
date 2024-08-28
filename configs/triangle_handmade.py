from configs.library import *
from utils import FeeRate

classifier = body_classifiers.custom
classifier.params.source_file = Param("data/test.csv", [None])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [None])
# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(3, [2, 3, 4])
# stops_processor.params.tp = Param(6, [4, 6, 8, 12])

config = EasyDict(
    wallet=Param(100, [None]),
    leverage=Param(1, [1]),
    date_start=Param("2010-01-01T00:00:00", ["2025-08-01"]),
    date_end=Param("2025-05-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.05, [0]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    allow_overturn=Param(False, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [32]),
    ticksize=Param(0.001, [None]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("D", ["H1"]),
    ticker=Param("TSLA", ["AAPL"]),
    data_type=Param("yahoo", ["metatrader"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(FeeRate(0, 0), [None]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
