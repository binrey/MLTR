from common.type import TimePeriod
from common.utils import update_config
from configs.tunnel import config

updates = {
    "ticker": "ETHUSDT",
    "ticksize": 0.01,
    "wallet": 100,
    "leverage": 10,
    "trailing_stop_rate": 0.003,
    "hist_buffer_size": 32,
    "period": TimePeriod.M60,
    "decision_maker": {
        "ncross": 4
    }
}

# Update the config dictionary
update_config(config, updates)