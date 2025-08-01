from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.volprof.base import backtest, bybit

updates = dict(
    symbol = Symbols.BTCUSDT,
    period=TimePeriod.M60,
    wallet=1000,
    volume_control = {"deposit_fraction": 0.5},
    leverage = 1,
    hist_size = 64,
    trailing_stop = {"rate": 0.02},
    decision_maker = {"sharpness": 4},
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)