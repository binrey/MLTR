from easydict import EasyDict

from experts import *

body_classifiers = {
    "trngl_simp": {
        "func": cls_triangle_simple,
        "params": dict(npairs=3)
        },
    "trend": dict( 
        func=cls_trend,
        params=dict(npairs=3)
        )    
}

stops_processors = {
    "stops_fixed": {
        "func": stops_fixed,
        "params": dict(
            tp=4, 
            sl=2
            )
    },
    "stops_dynamic": {
        "func": stops_dynamic,
        "params": dict()
    }    
}
# ----------------------------------------------------------------
# Test configuration
test_config = EasyDict(
    body_classifier=body_classifiers["trend"],
    stops_processor=stops_processors["stops_fixed"],
    wait_entry_point=9999,
    hist_buffer_size=20,
    tstart=0,
    tend=None,
    data_file="data/metatrader/M30/SBER_M30_200801091100_202309292330.csv",
    save_plots=False,
    backtest_metrics="max_profit"
)
# ----------------------------------------------------------------
# Optimization configuration
optim_config = EasyDict(
    body_classifier=body_classifiers,
    stops_processor=stops_processors,
    wait_entry_point=9999
)
