from easydict import EasyDict

from experts import *

class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    trngl_simp = EasyDict(
        func=ClsTriangleSimp,
        params=EasyDict(npairs=Param(3, [3]))
    ),
    trngl_comp = EasyDict(
        func=ClsTriangleComp,
        params=EasyDict(npairs=Param(3, [3]))
    ),
    trend = EasyDict( 
        func=ClsTrend,
        params=EasyDict(npairs=Param(2, [2]))
        )    
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(None, [None]), 
            sl=Param(4, [2, 4, 6])
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
bitfinex_list = ["BTCUSD", "ETHUSD"]#, "TRXUSD", "XRPUSD"]
yahoo_list = ["MSFT", "AMZN", "AAPL", "GOOG", "NFLX", "TSLA"]
moex_list = ["SBER", "ROSN", "LKOH", "GMKN", "GAZP"]
forts_list = ["SBRF", "ROSN", "LKOH", "GAZR"]

config = EasyDict(
    date_start=Param("2008-01-01", ["2008-01-01"]),
    trailing_stop_rate=Param(0.02, [0.01, 0.02, 0.04]),
    trailing_stop_type=Param(2, [1]),
    body_classifier=Param(body_classifiers["trend"], [body_classifiers[k] for k in ["trend"]]),
    stops_processor=Param(stops_processors["stops_fixed"], [stops_processors[k] for k in ["stops_fixed"]]),
    wait_entry_point=Param(9999, [9999]),
    hist_buffer_size=Param(90, [30]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["H1"]),
    ticker=Param("BTCUSD", bitfinex_list),
    data_type=Param("bitfinex", ["bitfinex"]),
    save_plots=Param(False, [False]),
)
