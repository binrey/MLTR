from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.volprof.base import backtest, bybit

updates = dict(
    wallet=100,
    symbol = Symbols.BTCUSDT,
    period=TimePeriod.M60,
    hist_size = 64,
    trailing_stop = {"rate": 0.02},
    decision_maker = {"sharpness": 4}
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)