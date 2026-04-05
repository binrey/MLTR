import numpy as np

from common.type import TimePeriod, Symbols
from common.utils import FeeRate

config = dict(
    date_start=np.datetime64("2017-01-01T00:00:00"),
    date_end=np.datetime64("2026-04-01T00:00:00"),
    period=TimePeriod.D,
    symbols=[Symbols.BTCUSDT],
    data_type="bybit",
    fee_rate=FeeRate(0.055, 0.00016),
    hist_size=200,
    epochs=10,
)

backtest = config