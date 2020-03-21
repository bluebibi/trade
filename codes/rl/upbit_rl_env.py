import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
import copy
from codes.rl.upbit_rl_utils import array_2d_to_dict_list_order_book, get_buying_price_by_order_book, \
    get_selling_price_by_order_book, get_rl_dataset, EnvironmentType, EnvironmentStatus, BuyerAction, SellerAction
from codes.rl.upbit_rl_constants import BUY_AMOUNT, WINDOW_SIZE, INITIAL_TOTAL_KRW


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

        # if self.env_type is EnvironmentType.TRAIN_VALID:
        #     self.x_train, self.x_train_base_datetime, self.train_size, \
        #     self.x_valid, self.x_valid_base_datetime,  self.valid_size = get_rl_dataset(self.coin_name, train_valid_split=True)

        if self.env_type is EnvironmentType.TRAIN_VALID:
            self.x, self.x_base_datetime, self.size = get_rl_dataset(self.coin_name, train_valid_split=False)

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
        self.total_steps = None
        self.account_history = None

        self.status = None

        init_str = "[COIN NAME: {0}] INIT\nOBSERVATION SPACE: {1}\nBUYER_ACTION SPACE: {2}\nSELLER_ACTION_SPACE: {3}" \
                   "\nRAW_DATA_SHAPE: {4}\nWINDOW_SIZE: {5}\n".format(
            self.coin_name,
            self.observation_space,
            self.buyer_action_space,
            self.seller_action_space,
            self.x.shape,
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
            self.data = self.x
            self.data_datetime = self.x_base_datetime
            self.data_size = self.size
        else:
            pass
            # raise ValueError("Problem at self.env_type : {0}".format(self.env_type))

        self.current_step = 0
        self.steps_left = self.data_size
        self.total_steps = self.data_size

        observation, info_dic = self._next_observation(next_env_status=EnvironmentStatus.TRYING_BUY)

        reset_str = "[COIN NAME: {0}] RESET\nENV_TYPE: {1}\nCURRENT_STEPS: {2}\nSTEPS_LEFT: {3}" \
                    "\nINITIAL_BALANCE: {4}\nINITIAL_TOTAL_PROFIT: {5}\nINITIAL_HOLD_COIN_QUANTITY: {6}" \
                    "\nBUY_AMOUNT: {7}won\nINITIAL OBSERVATION: {8}\nINITIAL_CHANGE_INDEX:{9}\nINITIAL_COIN_PRICE:{10}" \
                    "\nINITIAL_COIN_QUANTITY:{11}\nINITIAL_COMMISSION_FEE:{12}" \
                    "\nFIRST_DATETIME:{13}\nLAST_DATETIME:{14}\n".format(
            self.coin_name,
            self.env_type,
            self.current_step,
            self.steps_left,
            self.balance,
            self.total_profit,
            self.hold_coin_quantity,
            BUY_AMOUNT,
            observation.shape,
            info_dic["change_index"],
            info_dic["coin_unit_price"],
            info_dic["coin_quantity"],
            info_dic["commission_fee"],
            self.x_base_datetime[0],
            self.x_base_datetime[-1]
        )

        print(reset_str)

        self.current_step += 1
        self.steps_left -= 1

        return observation, info_dic

    def step_with_info_dic(self, action, info_dic):
        reward = None
        next_observation = next_info_dic = next_env_status = base_data = None

        if self.status is EnvironmentStatus.TRYING_BUY:
            if action is BuyerAction.BUY_HOLD:
                reward = 0.0
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
                reward = 0.0
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

                reward = float(profit) / 1000.0
                next_env_status = EnvironmentStatus.TRYING_BUY

        if self.steps_left == 0 or self.balance <= 0.0:
            done = True
        else:
            done = False
            next_observation, next_info_dic = self._next_observation(next_env_status=next_env_status)
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
            base_data = current_x[0]
        else:
            coin_krw, coin_unit_price, coin_quantity, commission_fee = get_selling_price_by_order_book(
                self.hold_coin_quantity, order_book_list
            )
            base_data = self.data[self.current_step - 1][-1]

        # mean_price_list = []
        # for j in range(WINDOW_SIZE):
        #     mean_price = (current_x[j][1] + current_x[j][11]) / 2
        #     mean_price_list.append(mean_price)
        #
        # cf = changefinder.ChangeFinderARIMA()
        # c = [cf.update(p) for p in mean_price_list]
        #
        # change_index = c[-1]

        current_x = current_x / base_data

        assert current_x.shape == self.observation_space.shape, \
               "current_x.shape: {0}, self.observation_space.shape: {1}".format(current_x.shape, self.observation_space.shape)

        assert type(coin_krw) is type(10), "Type mismatch"

        info_dic = {
            "change_index": 0.0,
            "coin_krw": coin_krw,
            "coin_unit_price": coin_unit_price,
            "coin_quantity": coin_quantity,
            "commission_fee": commission_fee
        }

        return current_x, info_dic
