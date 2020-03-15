import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from codes.rl.upbit_rl_constants import BUY_AMOUNT, MAX_EPISODES, \
    REPLAY_MEMORY_THRESHOLD_FOR_TRAIN, TRAIN_INTERVAL, QNET_COPY_TO_TARGET_QNET_INVERVAL

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import sys,os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
import changefinder
import copy
from codes.rl.upbit_rl_policy import DeepBuyerPolicy, DeepSellerPolicy
from common.global_variables import *
from codes.rl.upbit_rl_utils import array_2d_to_dict_list_order_book, get_buying_price_by_order_book, \
    get_selling_price_by_order_book, print_before_step, get_rl_dataset, print_after_step, EnvironmentType, \
    EnvironmentStatus, BuyerAction, SellerAction, draw_performance


class UpbitEnvironment(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, coin_name, env_type=EnvironmentType.TRAIN_VALID, serial=True):
        super(UpbitEnvironment, self).__init__()

        self.coin_name = coin_name
        self.buyer_action_space = spaces.Discrete(2)
        self.seller_action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(WINDOW_SIZE, 21), dtype=np.float16
        )
        self.env_type = env_type
        self.serial = serial

        if self.env_type is EnvironmentType.TRAIN_VALID:
            self.x_train, self.x_train_base_datetime, self.train_size, \
            self.x_valid, self.x_valid_base_datetime,  self.valid_size = get_rl_dataset(self.coin_name)

        self.balance = None
        self.total_profit = None
        self.hold_coin_krw = None
        self.hold_coin_quantity = None
        self.hold_coin_unit_price = None

        self.just_bought_coin_krw = None
        self.just_bought_coin_quantity = None
        self.just_bought_coin_unit_price = None

        self.just_sold_coin_krw = None
        self.just_sold_coin_quantity = None
        self.just_sold_coin_unit_price = None

        self.current_step = None
        self.steps_left = None
        self.account_history = None

        self.train = True
        self.status = None

        init_str = "[COIN NAME: {0}] INIT\nOBSERVATION SPACE: {1}\nBUYER_ACTION SPACE: {2}\nSELLER_ACTION_SPACE: {3}\nRAW_TRAIN_DATA_SHAPE: {4}" \
                   "\nRAW_VALID_DATA_SHAPE: {5}\nWINDOW_SIZE: {6}\n".format(
            self.coin_name,
            self.observation_space,
            self.buyer_action_space,
            self.seller_action_space,
            self.x_train.shape,
            self.x_valid.shape,
            WINDOW_SIZE
        )

        print(init_str)

    def reset(self):
        self.balance = INITIAL_TOTAL_KRW
        self.total_profit = 0.0
        self.hold_coin_krw = 0
        self.hold_coin_quantity = 0.0
        self.hold_coin_unit_price = 0.0
        self.just_bought_coin_krw = None
        self.just_bought_coin_quantity = None
        self.just_bought_coin_unit_price = None
        self.just_sold_coin_krw = None
        self.just_sold_coin_quantity = None
        self.just_sold_coin_unit_price = None

        self.status = EnvironmentStatus.TRYING_BUY

        if self.env_type == EnvironmentType.TRAIN_VALID:
            if self.train:
                self.data = self.x_train
                self.data_datetime = self.x_train_base_datetime
                self.data_size = self.train_size
            else:
                self.data = self.x_valid
                self.data_datetime = self.x_valid_base_datetime
                self.data_size = self.valid_size
        else:
            pass
            # raise ValueError("Problem at self.env_type : {0}".format(self.env_type))

        self.current_step = 0
        self.steps_left = self.data_size

        observation, info_dic = self._next_observation(next_env_status=EnvironmentStatus.TRYING_BUY)

        reset_str = "[COIN NAME: {0}] RESET\nENV_TYPE: {1}\nCURRENT_STEPS: {2}\nSTEPS_LEFT: {3}" \
                    "\nINITIAL_BALANCE: {4}\nINITIAL_TOTAL_PROFIT: {5}\nINITIAL_HOLD_COIN_QUANTITY: {6}" \
                    "\nBUY_AMOUNT: {7}won\nINITIAL OBSERVATION: {8}\nINITIAL_BASE_ASK_PRICE:{9}" \
                    "\nINITIAL_BASE_BID_PRICE:{10}\nINITIAL_CHANGE_INDEX:{11}\nINITIAL_COIN_PRICE:{12}" \
                    "\nINITIAL_COIN_QUANTITY:{13}\nINITIAL_COMMISSION_FEE:{14}" \
                    "\nTRAIN_FIRST_DATETIME:{15}\nTRAIN_LAST_DATETIME:{16}\nVALIDATION_FIRST_DATETIME:{16}" \
                    "\nVALIDATION_LAST_DATETIME:{15}\n".format(
            self.coin_name,
            self.env_type,
            self.current_step,
            self.steps_left,
            self.balance,
            self.total_profit,
            self.hold_coin_quantity,
            BUY_AMOUNT,
            observation.shape,
            info_dic["base_ask_price"],
            info_dic["base_bid_price"],
            info_dic["change_index"],
            info_dic["coin_unit_price"],
            info_dic["coin_quantity"],
            info_dic["commission_fee"],
            self.x_train_base_datetime[0],
            self.x_train_base_datetime[-1],
            self.x_valid_base_datetime[0],
            self.x_valid_base_datetime[-1]
        )

        print(reset_str)

        self.current_step += 1
        self.steps_left -= 1

        return observation, info_dic

    def step_with_info_dic(self, action, info_dic):
        reward = None
        next_observation = next_info_dic = None

        if self.status is EnvironmentStatus.TRYING_BUY:
            if action is BuyerAction.BUY_HOLD:
                reward = -0.1
                next_env_status = EnvironmentStatus.TRYING_BUY

            elif action is BuyerAction.MARKET_BUY:
                self.hold_coin_krw = info_dic["coin_krw"]
                self.hold_coin_quantity = info_dic["coin_quantity"]
                self.hold_coin_unit_price = info_dic["coin_unit_price"]
                self.balance -= self.hold_coin_krw

                self.just_bought_coin_krw = info_dic["coin_krw"]
                self.just_bought_coin_quantity = info_dic["coin_quantity"]
                self.just_bought_coin_unit_price = info_dic["coin_unit_price"]

                reward = "Pending"
                next_env_status = EnvironmentStatus.TRYING_SELL
        else:
            if action is SellerAction.SELL_HOLD:
                self.hold_coin_unit_price = info_dic["coin_unit_price"]
                self.hold_coin_krw = info_dic["coin_krw"]
                self.hold_coin_quantity = info_dic["coin_quantity"]
                reward = -0.1
                next_env_status = EnvironmentStatus.TRYING_SELL

            elif action is SellerAction.MARKET_SELL:
                sold_coin_krw = info_dic["coin_krw"]
                profit = sold_coin_krw - self.just_bought_coin_krw
                self.total_profit += profit
                self.balance += sold_coin_krw

                self.hold_coin_krw = 0
                self.hold_coin_quantity = 0.0
                self.hold_coin_unit_price = 0.0

                self.just_sold_coin_krw = info_dic["coin_krw"]
                self.just_sold_coin_quantity = info_dic["coin_quantity"]
                self.just_sold_coin_unit_price = info_dic["coin_unit_price"]

                reward = float(profit)
                next_env_status = EnvironmentStatus.TRYING_BUY

        done = True if self.steps_left == 0 else False
        if not done:
            next_observation, next_info_dic = self._next_observation(next_env_status)

        self.current_step += 1
        self.steps_left -= 1

        return next_observation, reward, done, next_info_dic

    def _next_observation(self, next_env_status):
        current_x = copy.deepcopy(self.data[self.current_step])

        order_book_list = array_2d_to_dict_list_order_book(current_x[-1])

        if next_env_status is EnvironmentStatus.TRYING_BUY:
            coin_krw, coin_unit_price, coin_quantity, commission_fee = get_buying_price_by_order_book(
                BUY_AMOUNT, order_book_list
            )
        else:
            coin_krw, coin_unit_price, coin_quantity, commission_fee = get_selling_price_by_order_book(
                self.hold_coin_quantity, order_book_list
            )

        base_ask_price = 0.0
        base_bid_price = 0.0
        mean_price_list = []
        for j in range(WINDOW_SIZE):
            mean_price = (current_x[j][1] + current_x[j][11]) / 2
            mean_price_list.append(mean_price)
            if j == 0:
                base_ask_price = current_x[j][1]
                base_bid_price = current_x[j][11]

        cf = changefinder.ChangeFinderARIMA()
        c = [cf.update(p) for p in mean_price_list]

        change_index = c[-1]

        current_x = current_x / current_x[0]

        assert current_x.shape == self.observation_space.shape, \
               "current_x.shape: {0}, self.observation_space.shape: {1}".format(current_x.shape, self.observation_space.shape)

        assert type(coin_krw) is type(10), "Type mismatch"

        info_dic = {
            "base_ask_price": base_ask_price,
            "base_bid_price": base_bid_price,
            "change_index": change_index,
            "coin_krw": coin_krw,
            "coin_unit_price": coin_unit_price,
            "coin_quantity": coin_quantity,
            "commission_fee": commission_fee
        }

        return current_x, info_dic


def main():
    coin_name = "ARK"

    env = UpbitEnvironment(coin_name=coin_name, env_type=EnvironmentType.TRAIN_VALID, serial=True)
    buyer_policy = DeepBuyerPolicy()
    seller_policy = DeepSellerPolicy()

    total_profit_list = []
    buyer_loss_list = []
    seller_loss_list = []

    for episode in range(MAX_EPISODES):
        done = False
        num_steps = 0
        observation, info_dic = env.reset()
        epsilon = max(0.001, 0.1 - 0.01 * (episode / 1000))
        buyer_loss = 0.0
        seller_loss = 0.0

        while not done:
            print_before_step(env, coin_name, episode, MAX_EPISODES, num_steps, info_dic)
            if env.status is EnvironmentStatus.TRYING_BUY:
                action = buyer_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is BuyerAction.MARKET_BUY:
                    action = 1
                    buyer_policy.pending_buyer_transition = [observation, action, None, None, None]
                    next_env_state = EnvironmentStatus.TRYING_SELL
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    buyer_policy.buyer_memory.put((observation, action, reward, next_observation, done_mask))
                    next_env_state = EnvironmentStatus.TRYING_BUY

            elif env.status is EnvironmentStatus.TRYING_SELL:
                action = seller_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is SellerAction.MARKET_SELL:
                    done_mask = 0.0 if done else 1.0

                    buyer_policy.pending_buyer_transition[2] = reward
                    buyer_policy.pending_buyer_transition[3] = next_observation
                    buyer_policy.pending_buyer_transition[4] = done_mask
                    buyer_policy.buyer_memory.put(tuple(buyer_policy.pending_buyer_transition))
                    buyer_policy.pending_buyer_transition = None

                    done_mask = 0.0 if done else 1.0
                    action = 1
                    seller_policy.seller_memory.put((observation, action, reward, next_observation, done_mask))
                    next_env_state = EnvironmentStatus.TRYING_BUY
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    seller_policy.seller_memory.put((observation, action, reward, next_observation, done_mask))
                    next_env_state = EnvironmentStatus.TRYING_SELL

            else:
                raise ValueError("Environment Status Error: {0}".format(env.status))

            num_steps += 1
            print_after_step(env, action, next_observation, reward, done, buyer_policy, seller_policy)

            # Replay Memory 저장 샘플이 충분하다면 buyer_policy 또는 seller_policy 강화학습 훈련 (딥러닝 모델 최적화)
            if num_steps % TRAIN_INTERVAL == 0:
                if buyer_policy.buyer_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                    buyer_loss = buyer_policy.train()
                if seller_policy.seller_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                    seller_loss = seller_policy.train()

            total_profit_list.append(env.total_profit)
            buyer_loss_list.append(buyer_loss)
            seller_loss_list.append(seller_loss)
            draw_performance(total_profit_list, buyer_loss_list, seller_loss_list)

            if num_steps % QNET_COPY_TO_TARGET_QNET_INVERVAL == 0:
                buyer_policy.qnet_copy_to_target_qnet()
                seller_policy.qnet_copy_to_target_qnet()

            # 다음 스텝 수행을 위한 사전 준비
            observation = next_observation
            info_dic = next_info_dic
            env.status = next_env_state

            if env.balance <= 0.0:
                done = True


if __name__ == "__main__":
    main()