from common.type import TimePeriod
from common.utils import update_config
from configs.volprof.BTCUSDT import backtest, bybit

updates = dict(
    volume_control = {"deposit_fraction": 0.5},
    period=TimePeriod.M1,
    wallet=1000,
    traid_stops_min_size = 0.025,
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)