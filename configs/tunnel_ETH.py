from configs.tunnel import Param
from configs.tunnel import config


config.ticker.test = "ETHUSDT"
# config.lot.test = 0.3
config.body_classifier.test.params.ncross.test = 0.8
config.trailing_stop_rate.test = 0.01
