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
        func=ClsTriangleSimp,
        params=EasyDict(npairs=Param(3, [3]))
        ),
    trend = EasyDict( 
        func=ClsTrend,
        params=EasyDict(npairs=Param(2, [2, 3]),
            maxdrop=Param(0.18, [0.01, 0.02, 0.04])
            )
        ),    
    tunnel = EasyDict( 
        func=ClsTunnel,
        params=EasyDict(
            ncross=Param(8, [4, 8, 16, 32, 64])
            )
        ) 
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(None, [None]), 
<<<<<<< HEAD
            sl=Param(2, [0.5, 1, 2, 4, 8])
=======
            sl=Param(2, [1, 2, 3, 5])
>>>>>>> binary_cls
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
    trailing_stop_rate=Param(0.005, [0.00125, 0.0025, 0.005, 0.01]),
    trailing_stop_type=Param(1, [1]),
<<<<<<< HEAD
    body_classifier=Param(body_classifiers["trend"], [body_classifiers[k] for k in ["trend"]]),
    stops_processor=Param(stops_processors["stops_fixed"], [stops_processors[k] for k in ["stops_fixed"]]),
    wait_entry_point=Param(9999, [9999]),
    hist_buffer_size=Param(34, [32]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["H1"]),
    ticker=Param("BTCUSDT", ["ETHUSDT"]),
=======
    body_classifier=Param(body_classifiers["tunnel"], [body_classifiers[k] for k in ["tunnel"]]),
    stops_processor=Param(stops_processors["stops_dynamic"], [stops_processors[k] for k in ["stops_dynamic"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(64, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("M15", ["M15"]),
    ticker=Param("BTCUSDT", ["BTCUSDT", "ETHUSDT"]),
>>>>>>> binary_cls
    data_type=Param("metatrader", ["metatrader"]),
    save_plots=Param(False, [False]),
    run_model_device=Param(None, [None])
)
