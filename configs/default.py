from easydict import EasyDict

from experts import *

class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    trngl_simp = EasyDict(
        func=ClsTriangleSimp,
        params=EasyDict(npairs=Param(3, [2, 3]))
    ),
    trend = EasyDict( 
        func=ClsTrend,
        params=EasyDict(npairs=Param(3, [2, 3]))
        )    
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(2, [2, 4, 8]), 
            sl=Param(2, [2, 4, 8])
            )
),
    stops_dynamic = EasyDict(
        func=StopsDynamic,
        params=EasyDict(
            tp_active=Param(False, [False]),
            sl_active=Param(True, [False])
            )
    )    
)
# ----------------------------------------------------------------
# Configuration
config = EasyDict(
    body_classifier=Param(body_classifiers["trend"], list(body_classifiers.values())),
    stops_processor=Param(stops_processors["stops_dynamic"], list(stops_processors.values())),
    wait_entry_point=Param(9999, [9999]),
    hist_buffer_size=Param(30, [30]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["H1"]),
    ticker=Param("BTCUSD", ["BTCUSD", "ETHUSD", "TRXUSD", "XRPUSD"]),
    data_type=Param("bitfinex", ["bitfinex"]),
    save_plots=Param(False, [False]),
)
