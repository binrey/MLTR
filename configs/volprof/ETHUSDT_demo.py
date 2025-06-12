from common.type import TimePeriod
from common.utils import update_config
from configs.volprof.ETHUSDT import backtest, bybit

updates = dict(
    decision_maker = {"sharpness": 0},
    wallet=500,
    period=TimePeriod.M1,
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)
