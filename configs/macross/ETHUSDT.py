from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.macross.base import backtest, bybit

updates = dict(
    symbol = Symbols.ETHUSDT,
    period=TimePeriod.M60,
    wallet=1000,
    volume_control = {"deposit_fraction": 1},
    leverage = 1,
    lot = 0.12,
    decision_maker = {
        "ma_fast_period": 8,
        "upper_levels": 0,
        "lower_levels": 100,
        "min_step": 0.1,
    },
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)