from common.type import Symbols
from common.utils import update_config
from configs.macross.base import backtest, bybit

updates = dict(
    symbol=Symbols.ETHUSDT,
    volume_control = {"deposit_fraction": 0.45},
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)