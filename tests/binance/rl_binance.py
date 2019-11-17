import sys, os

from stable_baselines import PPO2
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.live import CCXTExchange
from tensortrade.strategies import StableBaselinesTradingStrategy

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from rl_common import *

import ccxt  # noqa: E402

binance = ccxt.binance({
    'apiKey': API_KEY_BINANCE,
    'secret': SECRET_KEY_BINANCE,
    'enableRateLimit': True,
})

exchange = CCXTExchange(
    exchange=binance,
    timeframe=timeframe,
    base_instrument=base_instrument,
    should_pretransform_obs=True,
    observation_type="ohlcv"
)

environment = TradingEnvironment(
    exchange=exchange,
    action_strategy=action_strategy,
    reward_strategy=reward_strategy,
    feature_pipeline=feature_pipeline
)

print("environment.observation_space: {0}".format(environment.observation_space))
print("environment.action_space: {0}".format(environment.action_space))

strategy = StableBaselinesTradingStrategy(
    environment=environment,
    model=PPO2,
    params={
        "learning_rate": 1e-5
    }
)

strategy.environment = environment

strategy.restore_agent(path=os.path.join(PROJECT_HOME, "rl", "binance", "ppo_btc_{0}".format(timeframe)))

live_performance = strategy.run(episodes=0)

print(live_performance)

strategy.save_agent(path="./ppo_btc_1h")
