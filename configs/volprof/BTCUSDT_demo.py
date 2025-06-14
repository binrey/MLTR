from common.type import TimePeriod
from common.utils import update_config
from configs.volprof.BTCUSDT import backtest, bybit

updates = dict(
    volume_control = {"deposit_fraction": 0.5},
    decision_maker = {"sharpness": 1},
    period=TimePeriod.M1,
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)