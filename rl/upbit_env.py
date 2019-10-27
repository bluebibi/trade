import enum
from abc import ABC
from typing import Tuple

import gym
from gym import spaces
import numpy as np

ACTION_DESC = {
    0: "HOLD",
    1: "MARKET_BUY_25%",
    2: "MARKET_SELL_25%",
    3: "HOLD",
    4: "MARKET_BUY_50%",
    5: "MARKET_SELL_50%",
    6: "HOLD",
    7: "MARKET_BUY_75%",
    8: "MARKET_SELL_75%",
    9: "HOLD",
    10: "MARKET_BUY_100%",
    11: "MARKET_SELL_100%",
}


class EnvironmentType(enum.Enum):
    TRAINING = 0
    TESTING = 1
    LIVE = 2


class UpbitEnvironment(gym.Env):
    def __init__(self, coin_name, env_type=EnvironmentType.TRAINING):
        self.coin_name = coin_name
        self.action_space = spaces.Discrete(len(ACTION_DESC))
        self.observation_space = spaces.Box(
            low=np.array([[-2.0, -2.0]] * 125),
            high=np.array([[2.0, 2.0]] * 125),
            dtype=np.float32
        )
        self.env_type = env_type

    def step(self, action) -> Tuple[list, float, bool, dict]:
        pass

    def reset(self) -> list:
        pass

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    upbit_env = UpbitEnvironment("ADA")
    print(upbit_env.observation_space)
    state = upbit_env.reset()

    MAX_EPIOSDES = 10

    for episode in range(MAX_EPIOSDES):
        pass
