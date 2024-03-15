from easydict import EasyDict

from experts import *

class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    dummy = EasyDict( 
        func=ClsDummy,
        params=EasyDict()
            ),
    trngl_simp = EasyDict(
        func=ClsTriangle,
        params=EasyDict(npairs=Param(3, [3]))
        ),
    trend = EasyDict( 
        func=ClsTrend,
        params=EasyDict(npairs=Param(2, [2]),
            maxdrop=Param(0.18, [0.01])
            )
        ),    
    tunnel = EasyDict( 
        func=ClsTunnel,
        params=EasyDict(
            ncross=Param(17, [10, 15, 20, 25, 30, 35, 40])
            )
        ),
    custom = EasyDict( 
        func=ClsCustom,
        params=EasyDict(
            ncross=Param(0, [0])
            )
        ) 
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(None, [None]), 
            sl=Param(10, [0.25, 0.5, 1, 1.5])
            )
        ),
    stops_dynamic = EasyDict(
        func=StopsDynamic,
        params=EasyDict(
            tp_active=Param(False, [False]),
            sl_active=Param(True, [True])
            )
        )    
)
# ----------------------------------------------------------------
# Configuration

config = EasyDict(
    lot=Param(0.01, [0.01]),
    date_start=Param("2017-08-01", ["2017-08-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.007, [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(body_classifiers["trend"], [body_classifiers[k] for k in ["trend"]]),
    stops_processor=Param(stops_processors["stops_fixed"], [stops_processors[k] for k in ["stops_fixed"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(32, [32]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT"]),
    data_type=Param("bybit", ["bybit"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
