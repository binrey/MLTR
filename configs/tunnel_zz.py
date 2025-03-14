from common.utils import FeeRate
from configs.library import *

classifier = body_classifiers.tunzigzag
classifier.params.ncross = Param(5, [3, 5, 8, 12])
classifier.params.period = Param(3, [3, 5, 8])
classifier.params.nzz = Param(32, [32])

# stops_processor = stops_processors.stops_fixed
# stops_processor.params.sl = Param(2, [2, 3, 4])
# stops_processor.params.tp = Param(10, [1, 2, 3])

stops_processor = stops_processors.stops_dynamic
stops_processor.params.sl_active = Param(True, [True])
# stops_processor.params.tp_active = Param(True, [True])

config = EasyDict(
    wallet=Param(100, [100]),
    leverage=Param(1, [1]),
    date_start=Param("2000-09-01T00:00:00", ["2000-01-01"]),
    date_end=Param("2024-09-01", ["2025-01-01"]),
    no_trading_days=Param(set(), [set()]),
    rate=Param(0.006, [0.002, 0.004, 0.006]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(classifier, [classifier]),
    close_only_by_stops=Param(False, [False]),
    stops_processor=Param(stops_processor, [stops_processor]),
    wait_entry_point=Param(999, [999]),
    hist_size=Param(128, [128]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    
    period=Param("M60", ["M60"]),
    ticker=Param("BTCUSDT", ["BTCUSDT"]),
    equaty_step=Param(0.001, [0.001]),
    data_type=Param("metatrader", ["metatrader"]),
    
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None]),
    fee_rate=Param(FeeRate(0.1, 0.00016), [FeeRate(0.1, 0.00016)]),
    eval_buyhold=Param(True, [False]),
    fuse_buyhold=Param(False, [False]),
)
