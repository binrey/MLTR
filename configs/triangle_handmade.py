from common.utils import FeeRate
from configs.library import *

classifier = body_classifiers.custom
classifier.params.source_file = Param("data/test.csv", ["data/test.csv"])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(3, [2, 3, 4])
# stops_processor.params.tp = Param(6, [4, 6, 8, 12])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2010-01-01T00:00:00", ["2010-01-01"]),
    date_end=Param("2025-05-01", ["2025-05-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.16, [0.08, 0.12, 0.14, 0.16]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    close_only_by_stops=Param(False, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [8]),
    equaty_step=Param(0.001, [0.001]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("D", ["D"]),
    ticker=Param("TSLA", ["TSLA"]),
    data_type=Param("yahoo", ["yahoo"]),
    save_plots=Param(True, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(FeeRate(0, 0), [FeeRate(0, 0)]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
