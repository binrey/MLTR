from common.type import Symbols
from common.utils import update_config
from configs.random.base import backtest, bybit

updates = dict(
    symbol=Symbols.XRPUSDT,
    volume_control={"deposit_fraction": 1},
)

backtest = update_config(backtest, **updates)
bybit = update_config(bybit, **updates)