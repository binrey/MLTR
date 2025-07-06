from common.type import TimePeriod
from common.utils import update_config
from configs.volprof.ETHUSDT import backtest, bybit

updates = dict(
    volume_control = {"deposit_fraction": 0.45},
    period=TimePeriod.M5,
    traid_stops_min_size = 0.1,
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)
