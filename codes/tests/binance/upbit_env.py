from tensortrade.exchanges.simulated import FBMExchange
from rl_common import *
import pandas as pd

exchange = FBMExchange(
    timeframe=timeframe,
    base_instrument=base_instrument,
    should_pretransform_obs=True,
    times_to_generate=200000
)

exchange.reset()
df = exchange.data_frame

print(df)
print(df.describe())