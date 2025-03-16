from common.type import Symbols
from common.utils import update_config
from configs.vvol.base import config

updates = {
    "symbol": Symbols.BTCUSDT,
}

config = update_config(config, **updates)
