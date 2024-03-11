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
    trngl = EasyDict(
        func=ClsTriangle,
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
            ncross=Param(10, [5, 10, 20, 40])
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
            tp=Param(30, [2, 4, 8, 16]), 
            sl=Param(10, [2, 4, 8, 16])
            )
        ),
    stops_dynamic = EasyDict(
        func=StopsDynamic,
        params=EasyDict(
            tp_active=Param(True, [True]),
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
    lot=Param(0.01, [0.01]),
    date_start=Param("2005-01-01", ["2005-01-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate_long=Param(0.0, [0.00125, 0.0025, 0.005, 0.01]),
    trailing_stop_rate_shrt=Param(0.0, [0.001]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(body_classifiers["trngl"], [body_classifiers[k] for k in ["trngl"]]),
    stops_processor=Param(stops_processors["stops_dynamic"], [stops_processors[k] for k in ["stops_dynamic"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(128, [64]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("H1", ["H1"]),
    ticker=Param("AAPL", ["AAPL"]),
    data_type=Param("metatrader", ["metatrader"]),
    save_plots=Param(True, [False]),
    run_model_device=Param(None, [None])
)
