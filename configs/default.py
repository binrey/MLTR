from easydict import EasyDict

from experts import *

class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    trngl_simp = EasyDict(
        name="trngl_simp",
        func=cls_triangle_simple,
        params=EasyDict(npairs=Param(3, [2, 3]))
    ),
    trend = EasyDict( 
        name="trend",
        func=ClsTrend,
        params=EasyDict(npairs=Param(3, [2, 3]))
        )    
)

stops_processors = EasyDict(
    stops_fixed = {
        "name": "stops_fixed",
        "func": stops_fixed,
        "params": EasyDict(
            tp=Param(2, [1, 2]), 
            sl=Param(2, [1, 2])
            )
    },
    stops_dynamic = {
        "name": "stops_dynamic",
        "func": StopsDynamic,
        "params": EasyDict(dummy=Param(0, [0]))
    }    
)
# ----------------------------------------------------------------
# Configuration
config = EasyDict(
    body_classifier=Param(body_classifiers["trend"], list(body_classifiers.values())),
    stops_processor=Param(stops_processors["stops_dynamic"], list(stops_processors.values())),
    wait_entry_point=Param(9999, [9999]),
    hist_buffer_size=Param(40, [20]),
    tstart=Param(0, [0]),
    tend=Param(None, [1000]),
    period=Param("H1", ["H1"]),
    ticker=Param("BTCUSD", ["BTCUSD", "ETHUSD", "TRXUSD", "XRPUSD"]),
    data_type=Param("bitfinex", ["bitfinex"]),
    save_plots=Param(False, [False]),
    backtest_metrics=Param("max_profit", ["max_profit"])
,)
