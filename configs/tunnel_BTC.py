from common.type import TimePeriod
from configs.tunnel import config

config.update(dict(
ticker = "BTCUSDT",
ticksize = 0.001,

wallet = 100,
leverage = 1,

trailing_stop_rate = 0.005,
hist_buffer_size = 64,
period = TimePeriod.M60
))

config["decision_maker"]["ncross"] = 4