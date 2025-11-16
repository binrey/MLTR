from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.macross.base import backtest, bybit

updates = dict(
    symbol = Symbols.ETHUSDT,
    period=TimePeriod.M60,
    wallet=4000,
    volume_control = {"deposit_fraction": 0.25},
    leverage = 1,
    hist_size = 500,
    lot = 0.8,
    decision_maker = {
        "ma_fast_period": 80,
        "upper_levels": 0,
        "lower_levels": 1,
        "min_step": 0.6,
    },
    credentials="bybit_macross",
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)