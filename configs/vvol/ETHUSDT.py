from common.type import Symbols
from common.utils import update_config
from configs.vvol.base import config

updates = dict(
    symbol = Symbols.ETHUSDT,
    hist_size = 128,
    trailing_stop = {"rate": 0.01},
    decision_maker = {"sharpness": 3}
)

config = update_config(config, **updates)
