import random
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from upbit.upbit_api import Upbit

from common.global_variables import *

import enum

from typing import Tuple

import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

ACTIONS = {
    0: "HOLD",
    1: ("MARKET_BUY", 0.25),
    2: ("MARKET_SELL", 0.25),
    3: "HOLD",
    4: ("MARKET_BUY", 0.5),
    5: ("MARKET_SELL", 0.5),
    6: "HOLD",
    7: ("MARKET_BUY", 0.75),
    8: ("MARKET_SELL", 0.75),
    9: "HOLD",
    10: ("MARKET_BUY", 1.0),
    11: ("MARKET_SELL", 1.0),
}

BUY_AMOUNT = {
    1: ACTIONS[1][1],
    4: ACTIONS[4][1],
    7: ACTIONS[7][1],
    10: ACTIONS[10][1]
}

SELL_AMOUNT = {
    2: ACTIONS[2][1],
    5: ACTIONS[5][1],
    8: ACTIONS[8][1],
    11: ACTIONS[11][1]
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
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(WINDOW_SIZE, 130), dtype=np.float16
        )
        self.env_type = env_type
        self.serial = serial

        upbit_order_book_data = UpbitOrderBookBasedData(self.coin_name)
        self.x, self.x_train_normalized, self.train_size, self.x_valid_normalized, self.valid_size =\
            upbit_order_book_data.get_rl_dataset()

        self.balance = None

        self.last_net_worth = None
        self.net_worth = None

        self.coin_held = None

        self.current_step = None
        self.steps_left = None
        self.account_history = None

        self.current_x = None

        init_str = "[COIN NAME: {0}] INIT - OBSERVATION SPACE: {1}, ACTION SPACE: {2}, TRAIN_SIZE: {3}" \
                   ", VALID_SIZE: {4}, WINDOW_SIZE: {5}".format(
            self.coin_name,
            self.observation_space,
            self.action_space,
            self.train_size,
            self.valid_size,
            WINDOW_SIZE
        )

        print(init_str)


    def _reset_session(self):
        self.current_step = 0

        if self.env_type == EnvironmentType.TRAINING:
            self.data = self.x_train_normalized
            self.data_size = self.train_size
        elif self.env_type == EnvironmentType.VALIDATION:
            self.data = self.x_valid_normalized
            self.data_size = self.valid_size
        else:
            raise ValueError("Problem at self.env_type : {0}".format(self.env_type))

        if self.serial:
            self.steps_left = self.data_size - 1
            self.current_step = 0
        else:
            self.steps_left = np.random.randint(low=1, high=MAX_TRADING_SESSION)
            self.current_step = np.random.randint(low=0, high=self.data_size - self.steps_left)

    def reset(self) -> np.ndarray:
        self.last_net_worth = INITIAL_TOTAL_KRW
        self.balance = INITIAL_TOTAL_KRW
        self.net_worth = INITIAL_TOTAL_KRW
        self.coin_held = 0

        self._reset_session()

        self.account_history = np.repeat(
            [[self.net_worth], [0], [0], [0], [0]],
            repeats=WINDOW_SIZE,
            axis=1
        )

        self.trades = []

        reset_str = "[COIN NAME: {0}] RESET - ENV_TYPE: {1}, CURRENT_STEPS: {2}, STEPS_LEFT: {3}, INITIAL_NET_WORTH: {4}" \
                    ", INITIAL_BALANCE: {5}, ACCOUNT_HISTORY: {6}".format(
            self.coin_name,
            self.env_type,
            self.current_step,
            self.steps_left,
            self.net_worth,
            self.balance,
            self.account_history.shape
        )

        print(reset_str)
        return self._next_observation()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        current_price = self._take_action(action)
        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.balance += self.coin_held * current_price
            self.coin_held = 0
            self._reset_session()
            observation = None
            reward = None
            done = True
            info = {
                "steps_left": 0,
                "current_price": current_price,
                "net_worth": self.net_worth
            }
        else:
            observation = self._next_observation()
            reward = self._get_reward(action)
            done = self.net_worth <= 0
            info = {
                "steps_left": self.steps_left,
                "current_price": current_price,
                "net_worth": self.net_worth
            }
        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def _next_observation(self):
        scaled_history = self.scaler.fit_transform(self.account_history)
        obs = np.append(self.data[self.current_step], np.transpose(scaled_history), axis=1)
        self.current_x = self.x[self.current_step]

        self.current_step += 1

        assert obs.shape == self.observation_space.shape

        return obs

    def _get_reward(self, action):
        if action in [0, 3, 6, 9]:
            return 1
        elif action in [1, 4, 7, 10]:
            return 2
        elif action in [2, 5, 8, 11]:
            return self.net_worth - self.last_net_worth
        else:
            raise ValueError("_get_reward: {0}".format(action))

    def _take_action(self, action):
        coin_bought = 0
        coin_sold = 0
        cost = 0
        sales = 0

        if action in [0, 3, 6, 9]:
            last_ask_price = self.current_x[-1][1]
            last_bid_price = self.current_x[-1][61]
            calc_price = (last_ask_price + last_bid_price) / 2.0

        elif action in [1, 4, 7, 10]:
            buy_amount = BUY_AMOUNT[action]

            if self.env_type == EnvironmentType.LIVE:
                _, fee, calc_price, coin_bought = 0.0, 0.0, 0.0, 0.0
            else:
                _, fee, calc_price, coin_bought = upbit.get_expected_buy_coin_price_for_krw_and_ask_list(
                    ask_price_lst=self._get_ask_price_lst(),
                    ask_size_lst=self._get_ask_size_lst(),
                    krw=self.balance * buy_amount,
                    transaction_fee_rate=TRANSACTION_FEE_RATE
                )

            cost = coin_bought * calc_price + fee
            self.coin_held += coin_bought
            self.balance -= cost

            self.trades.append({
                'step': self.current_step,
                'amount': coin_bought,
                'total': cost,
                'type': "buy"
            })

        elif action in [2, 5, 8, 11]:
            sell_amount = SELL_AMOUNT[action]

            if self.env_type == EnvironmentType.LIVE:
                _, calc_price, fee, calc_krw_sum = 0.0, 0.0, 0.0, 0.0
            else:
                _, calc_price, fee, calc_krw_sum = upbit.get_expected_sell_coin_price_for_volume_and_bid_list(
                    bid_price_lst=self._get_bid_price_lst(),
                    bid_size_lst=self._get_bid_size_lst(),
                    volume=self.coin_held * sell_amount,
                    transaction_fee_rate=TRANSACTION_FEE_RATE
                )

            coin_sold = self.coin_held * sell_amount
            sales = calc_krw_sum + fee
            self.coin_held -= coin_sold
            self.balance += sales

            self.trades.append({
                'step': self.current_step,
                'amount': coin_sold,
                'total': sales,
                'type': "sell"
            })
        else:
            raise ValueError("_take_action: {0}".format(action))

        current_price = calc_price
        self.last_net_worth = self.net_worth
        self.net_worth = self.balance + self.coin_held * current_price

        self.account_history = np.append(
            self.account_history,
            [[self.net_worth], [coin_bought], [cost], [coin_sold], [sales]],
            axis=1
        )

        print("STEP: self.account_history: {0}".format(self.account_history.shape))

        return current_price

    def _get_ask_price_lst(self):
        ask_price_lst = []
        for i in range(15):
            ask_price_lst.append(self.current_x[-1][1 + 4 * i])
        return ask_price_lst

    def _get_ask_size_lst(self):
        ask_size_lst = []
        for i in range(15):
            ask_size_lst.append(self.current_x[-1][3 + 4 * i])
        return ask_size_lst

    def _get_bid_price_lst(self):
        bid_price_lst = []
        for i in range(15):
            bid_price_lst.append(self.current_x[-1][61 + 4 * i])
        return bid_price_lst

    def _get_bid_size_lst(self):
        bid_size_lst = []
        for i in range(15):
            bid_size_lst.append(self.current_x[-1][63 + 4 * i])
        return bid_size_lst


class RandomPolicy:
    def action(self, observation):
        return random.randint(0, len(ACTIONS))


def main():
    upbit_env = UpbitEnvironment(coin_name="ADA", env_type=EnvironmentType.TRAINING, serial=False)
    policy = RandomPolicy()

    observation = upbit_env.reset()
    print("observation", observation)
    print("x", upbit_env.x)

    MAX_EPIOSDES = 10

    for episode in range(MAX_EPIOSDES):
        action = policy.action(observation)
        next_observation, reward, done, info = upbit_env.step(action)
        print(next_observation, action, reward, done, info)

        observation = next_observation


if __name__ == "__main__":
    main()