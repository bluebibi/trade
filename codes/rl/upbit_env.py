import warnings
warnings.filterwarnings("ignore")

import random
import sys,os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import enum
from typing import Tuple
import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
import changefinder

from codes.upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from codes.upbit.upbit_api import Upbit
from common.global_variables import *


upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


class BuyerAction(enum.Enum):
    HOLD = 0,
    MARKET_BUY = 1


class SellerAction(enum.Enum):
    HOLD = 0,
    MARKET_SELL = 1


class EnvironmentType(enum.Enum):
    TRAIN_VALID = 0
    LIVE = 1


BUY_AMOUNT = 100000


class UpbitEnvironment(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, coin_name, env_type=EnvironmentType.TRAIN_VALID, serial=True):
        super(UpbitEnvironment, self).__init__()

        self.coin_name = coin_name
        self.buyer_action_space = spaces.Discrete(len(BuyerAction))
        self.seller_action_space = spaces.Discrete(len(SellerAction))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(WINDOW_SIZE, 14), dtype=np.float16
        )
        self.env_type = env_type
        self.serial = serial

        if self.env_type is EnvironmentType.TRAIN_VALID:
            upbit_order_book_data = UpbitOrderBookBasedData(self.coin_name)
            self.x_train_normalized, self.train_size, self.x_valid_normalized, self.valid_size = upbit_order_book_data.get_rl_dataset()

        self.balance = None

        self.last_net_worth = None
        self.net_worth = None

        self.coin_held = None

        self.current_step = None
        self.steps_left = None
        self.account_history = None

        self.current_x = None

        self.train = True

        init_str = "[COIN NAME: {0}] INIT\nOBSERVATION SPACE: {1}\nBUYER_ACTION SPACE: {2}\nSELLER_ACTION_SPACE: {3}\nRAW_TRAIN_DATA_SHAPE: {4}" \
                   "\nRAW_VALID_DATA_SHAPE: {5}\nWINDOW_SIZE: {6}\n".format(
            self.coin_name,
            self.observation_space,
            self.buyer_action_space,
            self.seller_action_space,
            self.x_train_normalized.shape,
            self.x_valid_normalized.shape,
            WINDOW_SIZE
        )

        print(init_str)


    def _reset_session(self):
        self.current_step = 0

        if self.env_type == EnvironmentType.TRAIN_VALID:
            if self.train:
                self.data = self.x_train_normalized
                self.data_size = self.train_size
            else:
                self.data = self.x_valid_normalized
                self.data_size = self.valid_size
        else:
            pass
            # raise ValueError("Problem at self.env_type : {0}".format(self.env_type))

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

        reset_str = "[COIN NAME: {0}] RESET\nENV_TYPE: {1}\nCURRENT_STEPS: {2}\nSTEPS_LEFT: {3}\nINITIAL_NET_WORTH: {4}" \
                    "\nINITIAL_BALANCE: {5}\nACCOUNT_HISTORY: {6}\nBUY_AMOUNT: {7}won\n".format(
            self.coin_name,
            self.env_type,
            self.current_step,
            self.steps_left,
            self.net_worth,
            self.balance,
            self.account_history.shape,
            BUY_AMOUNT
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
        mean_price = (self.data[self.current_step][:, 0] + self.data[self.current_step][:, 6]) / 2

        cf = changefinder.ChangeFinderARIMA()
        change_index = [cf.update(p) for p in mean_price]

        mean_price = np.expand_dims(mean_price, axis=1)
        change_index = np.expand_dims(change_index, axis=1)

        aux = np.append(mean_price, change_index, axis=1)
        obs = np.append(self.data[self.current_step], aux, axis=1)
        self.current_x = self.data[self.current_step]

        self.current_step += 1

        assert obs.shape == self.observation_space.shape, \
               "obs.shape: {0}, self.observation_space.shape: {1}".format(obs.shape, self.observation_space.shape)

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

        if action in [BuyerAction.HOLD, SellerAction.HOLD]:
            last_ask_price = self.current_x[-1][1]
            last_bid_price = self.current_x[-1][61]
            calc_price = (last_ask_price + last_bid_price) / 2.0

        elif action is BuyerAction.MARKET_BUY:
            buy_amount = BUY_AMOUNT

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

        elif action is SellerAction.MARKET_SELL:
            sell_amount = 10

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
        return random.randint(0, 1)


def main():
    upbit_env = UpbitEnvironment(coin_name="ARK", env_type=EnvironmentType.TRAIN_VALID, serial=True)
    policy = RandomPolicy()

    observation = upbit_env.reset()
    print("observation", observation)

    MAX_EPIOSDES = 10

    for episode in range(MAX_EPIOSDES):
        action = policy.action(observation)
        next_observation, reward, done, info = upbit_env.step(action)
        print(next_observation, action, reward, done, info)

        observation = next_observation


if __name__ == "__main__":
    main()