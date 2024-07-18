from configs.tunnel import Param
from configs.tunnel import config


config.ticker.test = "ETHUSDT"
config.ticksize.test = 0.01

config.wallet.test = 50
config.leverage.test = 1

config.body_classifier.test.params.ncross.test = 0
config.trailing_stop_rate.test = 0.005
config.period.test = "M1"