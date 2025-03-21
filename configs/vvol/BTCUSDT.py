from common.type import Symbols
from common.utils import update_config
from configs.vvol.base import config

updates = dict(
    symbol = Symbols.BTCUSDT,
    hist_size = 64,
    trailing_stop = {"rate": 0.02},
    decision_maker = {"sharpness": 4}
)

config = update_config(config, **updates)
