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
    wait_entry_point=20
)
# ----------------------------------------------------------------
# Optimization configuration
optim_config = EasyDict(
    body_classifier=body_classifiers,
    stops_processor=stops_processors,
    wait_entry_point=9999
)
