from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.volprof.base import backtest, trading

updates = dict(
    wallet=50,
    symbol = Symbols.ETHUSDT,
    period=TimePeriod.M60,
    hist_size = 128,
    trailing_stop = {"rate": 0.01},
    decision_maker = {"sharpness": 3}
)

backtest = update_config(backtest, **updates)
trading = update_config(trading, **updates)
