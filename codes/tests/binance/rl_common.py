from tensortrade.actions import DiscreteActionStrategy
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.rewards import SimpleProfitStrategy

timeframe = '1h'

symbol = 'ETH/BTC'
base_instrument = 'BTC'

# each ohlcv candle is a list of [ timestamp, open, high, low, close, volume ]

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(
    difference_order=0.6,
    inplace=True
)

feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='ETH/BTC')
