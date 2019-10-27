import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *

import enum

from typing import Tuple

import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData

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
    VALIDATION = 1
    LIVE = 2


class UpbitEnvironment(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, coin_name, env_type=EnvironmentType.TRAINING, serial=True):
        super(UpbitEnvironment, self).__init__()

        self.coin_name = coin_name
        self.action_space = spaces.Discrete(len(ACTION_DESC))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(130, WINDOW_SIZE + 1), dtype=np.float16
        )
        self.env_type = env_type
        self.serial = serial

        upbit_order_book_data = UpbitOrderBookBasedData(self.coin_name)
        self.x_train_normalized, self.train_size, self.x_valid_normalized, self.valid_size =\
            upbit_order_book_data.get_rl_dataset()

        init_str = "[COIN NAME: {0}] OBSERVATION SPACE: {1}, ACTION SPACE: {2}, TRAIN_SIZE: {3}" \
                   ", VALID_SIZE: {4}, WINDOW_SIZE: {5}".format(
            self.coin_name,
            self.observation_space,
            self.action_space,
            self.train_size,
            self.valid_size,
            WINDOW_SIZE
        )

        self.balance = None
        self.net_worth = None
        self.coin_held = None
        self.current_step = None
        self.steps_left = None
        self.account_history = None

        if self.env_type == EnvironmentType.TRAINING:
            self.data = self.x_train_normalized
            self.data_size = self.train_size
        elif self.env_type == EnvironmentType.VALIDATION:
            self.data = self.x_valid_normalized
            self.data_size = self.valid_size
        else:
            self.data = None
            self.data_size = None

        print(init_str)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    def reset(self) -> np.ndarray:
        self.balance = INITIAL_TOTAL_KRW
        self.net_worth = INITIAL_TOTAL_KRW
        self.coin_held = 0
        self.current_step = 0

        if self.serial:
            self.steps_left = self.data_size - 1
            self.current_step = 0
        else:
            self.steps_left = np.random.randint(low=1, high=MAX_TRADING_SESSION)
            self.current_step = np.random.randint(low=0, high=self.data_size - self.steps_left)

        self.account_history = np.repeat(
            [[self.net_worth], [0], [0], [0], [0]],
            repeats=WINDOW_SIZE,
            axis=1
        )

        self.trades = []

        reset_str = "[COIN NAME: {0}] ENV_TYPE: {1}, CURRENT_STEPS: {2}, STEPS_LEFT: {3}".format(
            self.coin_name,
            self.env_type,
            self.current_step,
            self.steps_left
        )

        print(reset_str)
        return self._next_observation()

    def render(self, mode='human'):
        pass

    def _next_observation(self):
        scaled_history = self.scaler.fit_transform(self.account_history)
        obs = np.append(self.data[self.current_step], np.transpose(scaled_history), axis=1)
        self.current_step += 1
        return obs


if __name__ == "__main__":
    upbit_env = UpbitEnvironment(coin_name="ADA", env_type=EnvironmentType.TRAINING, serial=False)
    state = upbit_env.reset()

    print(state)

    MAX_EPIOSDES = 10

    for episode in range(MAX_EPIOSDES):
        pass
