from easydict import EasyDict

from experts import *


class Param:
    def __init__(self, test, optim):
        self.test = test
        self.optim = optim
        

body_classifiers = EasyDict(
    # dummy = EasyDict( 
    #     func=ClsDummy,
    #     params=EasyDict()
    #         ),
    
    # trngl_simp = EasyDict(
    #     func=ClsTriangle,
    #     params=EasyDict(
    #         npairs=Param(None, [None]))
    #     ),
    # trend = EasyDict( 
    #     func=ClsTrend,
    #     params=EasyDict(
    #         npairs=Param(None, [None]),
    #         maxdrop=Param(None, [None])
    #         )
    #     ),    
    tunzigzag = EasyDict( 
        func=ClsTunZigZag,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ), 
    zigzag = EasyDict( 
        func=ClsZigZag,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ), 
    tunnel = EasyDict( 
        func=ClsTunnel,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ),
    tunnel2 = EasyDict( 
        func=ClsTunZigZag,
        params=EasyDict(
            ncross=Param(None, [None])
            )
        ),
    # bb = EasyDict( 
    #     func=ClsBB,
    #     params=EasyDict()
    #     ),
    # levels = EasyDict(
    #     func=ClsLevels,
    #     params=EasyDict()
    #     ),
    custom = EasyDict( 
        func=FileReader,
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