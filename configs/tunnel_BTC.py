from configs.tunnel import Param
from configs.tunnel import config


config.ticker.test = "BTCUSDT"
config.ticksize.test = 0.001

config.wallet.test = 100
config.leverage.test = 3

config.body_classifier.test.params.ncross.test = 4
config.trailing_stop_rate.test = 0.005
config.period.test = "M60"
