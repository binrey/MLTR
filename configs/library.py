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
        func=ClsTriangle,
        params=EasyDict(npairs=Param(3, [3]))
        ),
    trend = EasyDict( 
        func=ClsTrend,
        params=EasyDict(npairs=Param(None, [None]),
            maxdrop=Param(None, [None])
            )
        ),    
    tunnel = EasyDict( 
        func=ClsTunnel,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ),
    custom = EasyDict( 
        func=ClsCustom,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ) 
)

stops_processors = EasyDict(
    stops_fixed = EasyDict(
        func=StopsFixed,
        params=EasyDict(
            tp=Param(None, [None]), 
            sl=Param(None, [None])
            )
        ),
    stops_dynamic = EasyDict(
        func=StopsDynamic,
        params=EasyDict(
            tp_active=Param(False, [False]),
            sl_active=Param(False, [False])
            )
        )    
)