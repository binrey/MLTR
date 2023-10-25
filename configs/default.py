from easydict import EasyDict

from experts import *

class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    trngl_simp = dict(
        func=cls_triangle_simple,
        params=dict(npairs=Param(3, [2, 3]))
    ),
    trend = dict( 
        func=cls_trend,
        params=dict(npairs=Param(3, [2, 3]))
        )    
)

stops_processors = EasyDict(
    stops_fixed = {
        "func": stops_fixed,
        "params": dict(
            tp=Param(2, [1, 2, 4]), 
            sl=Param(2, [1, 2, 4])
            )
    },
    stops_dynamic = {
        "func": stops_dynamic,
        "params": dict()
    }    
)
# ----------------------------------------------------------------
# Test configuration
test_config = EasyDict(
    body_classifier=body_classifiers["trend"],
    stops_processor=stops_processors["stops_fixed"],
    wait_entry_point=9999,
    hist_buffer_size=20,
    tstart=0,
    tend=None,
    period="M30",
    ticker="SBER",
    data_type="metatrader",
    save_plots=False,
    backtest_metrics="max_profit"
)
# ----------------------------------------------------------------
# Optimization configuration
optim_config = EasyDict(
    body_classifier=Param(body_classifiers["trend"], list(body_classifiers.values())),
    stops_processor=Param(stops_processors["stops_fixed"], [stops_processors["stops_fixed"]]),
    wait_entry_point=Param(9999, [9999]),
    hist_buffer_size=Param(20, [20]),
    tstart=Param(0, [0]),
    tend=Param(5000, [None]),
    period=Param("M30", ["M30"]),
    ticker=Param("SBER", ["SBER", "GAZP"]),
    data_type=Param("metatrader", ["metatrader"]),
    save_plots=Param(False, [False]),
    backtest_metrics=Param("max_profit", ["max_profit"])
,)
