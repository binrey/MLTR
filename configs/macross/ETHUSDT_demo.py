from common.type import TimePeriod, Symbols
from common.utils import update_config
from configs.macross.base import backtest, bybit
from experts.ma_cross import ClsMACross

updates = dict(
    wallet=1000,
    symbol=Symbols.ETHUSDT,
    period=TimePeriod.M1,
    hist_size=64,
    decision_maker=dict(
        type=ClsMACross,
        mode = "trend",
        ma_fast_period=16,
        upper_levels = 1,
        lower_levels = 1,
        min_step=0.,
        speed=0.5
    ),    
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)