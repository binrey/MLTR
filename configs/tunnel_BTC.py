from common.type import TimePeriod
from common.utils import update_config
from configs.tunnel import config

updates = {
    "ticker": "BTCUSDT",
    "equaty_step": 0.001,
    "wallet": 100,
    "leverage": 1,
    "rate": 0.005,
    "hist_size": 64,
    "period": TimePeriod.M60,
    "decision_maker": {
        "ncross": 4
    }
}

update_config(config, updates)