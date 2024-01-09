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
        ),    
    saw = EasyDict( 
        func=ClsSaw,
        params=EasyDict(
            ncross=Param(3, [3, 5, 8, 13]),
            percentile=Param(10, [4, 8, 16])
            )
        ) 
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(None, [None]), 
            sl=Param(3, [5])
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
bitfinex_list = ["BTCUSD", "ETHUSD"]#, "TRXUSD"]#, "XRPUSD"]
yahoo_list = ["MSFT", "AMZN", "AAPL", "GOOG", "NFLX", "TSLA"]
moex_list = ["SBER", "ROSN", "LKOH", "GMKN", "GAZP"]
forts_list = ["SBRF", "ROSN", "LKOH", "GAZR"]

config = EasyDict(
    date_start=Param("2010-01-01", ["2010-01-01"]),
    date_end=Param("2024-01-01", ["2024-01-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0.005, [0.0025, 0.005, 0.01, 0.02, 0.03]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(body_classifiers["saw"], [body_classifiers[k] for k in ["saw"]]),
    stops_processor=Param(stops_processors["stops_dynamic"], [stops_processors[k] for k in ["stops_dynamic"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [128]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["H1"]),
    ticker=Param("BTCUSD", bitfinex_list),
    data_type=Param("metatrader", ["metatrader"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
