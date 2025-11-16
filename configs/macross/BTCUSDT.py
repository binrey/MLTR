from common.type import Symbols, TimePeriod
from common.utils import update_config
from configs.macross.base import backtest, bybit

updates = dict(
    symbol = Symbols.BTCUSDT,
    period=TimePeriod.M60,
    wallet=4000,
    volume_control = {"deposit_fraction": 0.25},
    leverage = 1,
    hist_size = 200,
    lot = 1, # 0.16, 0.12 0.1,
    decision_maker = {
        "ma_fast_period": 40, #16, 8 16,
        "upper_levels": 0, #0, 0 3,
        "lower_levels": 1, #100, 100 10,
        "min_step": 0.9, #0.3 0.3 0.25,
    },
    credentials="bybit_macross",
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)