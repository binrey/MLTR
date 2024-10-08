from common.type import TimePeriod
from configs.tunnel import Param, config

config.ticker.test = "ETHUSDT"
config.ticksize.test = 0.01

config.wallet.test = 30
config.leverage.test = 1

config.body_classifier.test.params.ncross.test = 4
config.trailing_stop_rate.test = 0.1
config.hist_buffer_size.test = 32
config.period.test = TimePeriod.M5