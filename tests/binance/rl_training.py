from stable_baselines import PPO2
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.strategies import StableBaselinesTradingStrategy

from training_stopper import TrainingStopper
from rl_common import *

exchange = FBMExchange(
    timeframe=timeframe,
    base_instrument=base_instrument,
    should_pretransform_obs=True
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

training_stopper = TrainingStopper(strategy=strategy, timeframe=timeframe)
performance = strategy.run(episodes=10, episode_callback=training_stopper)

