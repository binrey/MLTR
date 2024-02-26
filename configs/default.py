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
            ncross=Param(17, [15, 17, 20, 24])
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
    lot=Param(0.01, [0.01]),
    date_start=Param("2010-04-01", ["2010-01-01"]),
    date_end=Param("2024-03-01", ["2024-03-01"]),
    no_trading_days=Param(set(), [set()]),
    trailing_stop_rate=Param(0., [0.]),
    trailing_stop_type=Param(1, [1]),
    body_classifier=Param(body_classifiers["custom"], [body_classifiers[k] for k in ["custom"]]),
    stops_processor=Param(stops_processors["stops_fixed"], [stops_processors[k] for k in ["stops_fixed"]]),
    wait_entry_point=Param(999, [999]),
    hist_buffer_size=Param(256, [32]),
    tstart=Param(0, [0]),
    tend=Param(None, [None]),
    period=Param("D", ["D"]),
    ticker=Param("TSLA", ["TSLA"]),
    data_type=Param("yahoo", ["yahoo"]),
    save_plots=Param(True, [False]),
    run_model_device=Param(None, [None])
)
