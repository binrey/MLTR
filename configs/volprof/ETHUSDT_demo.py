from common.type import TimePeriod
from common.utils import update_config
from configs.volprof.ETHUSDT import backtest, bybit

updates = dict(
    hist_size = 32,
    trailing_stop = {"rate": 0.1},
    decision_maker = {"sharpness": 0},
    wallet=1000,
    period=TimePeriod.M1,
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)
