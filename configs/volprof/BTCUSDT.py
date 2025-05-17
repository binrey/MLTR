from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.volprof.base import backtest, bybit

updates = dict(
    wallet=1000,
    symbol = Symbols.BTCUSDT,
    period=TimePeriod.M15,
    hist_size = 256,
    trailing_stop = {"rate": 0.01},
    decision_maker = {"sharpness": 6}
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)