import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import sys,os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from gym import spaces
import numpy as np
import pandas as pd
from sklearn import preprocessing
import copy

from codes.rl.upbit_rl_utils import array_2d_to_dict_list_order_book, get_buying_price_by_order_book, \
    get_selling_price_by_order_book, EnvironmentType, EnvironmentStatus, BuyerAction, SellerAction
from codes.rl.upbit_rl_constants import BUY_AMOUNT, WINDOW_SIZE, INITIAL_TOTAL_KRW, SIZE_OF_FEATURE_WITHOUT_VOLUME, \
    SIZE_OF_FEATURE, FEATURES, FEATURES_WITHOUT_VOLUME, MAX_EPISODES
from web.db.database import naver_order_book_session, get_order_book_class


class UpbitEnvironment:
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, coin_name, args, env_type=EnvironmentType.TRAIN_VALID):
        self.coin_name = coin_name
        self.args = args

        self.buyer_action_space = spaces.Discrete(2)
        self.seller_action_space = spaces.Discrete(2)

        if self.args.volume:
            self.input_size = SIZE_OF_FEATURE
        else:
            self.input_size = SIZE_OF_FEATURE_WITHOUT_VOLUME

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(WINDOW_SIZE, self.input_size), dtype=np.float16
        )
        self.env_type = env_type

        self.data = None
        self.data_datetime = None
        self.data_size = None

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
                   "\nWINDOW_SIZE: {4}\n".format(
            self.coin_name,
            self.observation_space,
            self.buyer_action_space,
            self.seller_action_space,
            WINDOW_SIZE
        )

        print(init_str)

    def reset(self, episode, epsilon, buyer_policy, seller_policy):
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

        if self.env_type is EnvironmentType.TRAIN_VALID:
            self.data, self.data_datetime, self.data_size = self.get_rl_dataset(self.coin_name, self.args, train_valid_split=False)
        else:
            pass
            # raise ValueError("Problem at self.env_type : {0}".format(self.env_type))

        self.current_step = 0
        self.steps_left = self.data_size
        self.total_steps = self.data_size

        observation, info_dic = self._next_observation(next_env_status=EnvironmentStatus.TRYING_BUY)

        reset_str = "\n[COIN NAME: {0}] RESET\nENV_TYPE: {1}\nEPISODE/MAX_EPISODES: {2}/{3}\nCURRENT_STEPS/TOTAL_STEPS: {4}/{5}" \
                    "\nINITIAL_BALANCE: {6}\nINITIAL_TOTAL_PROFIT: {7}\nINITIAL_HOLD_COIN_QUANTITY: {8}" \
                    "\nBUY_AMOUNT: {9} won\nOBSERVATION SHAPE: {10}\nFIRST_DATETIME: {11}\nLAST_DATETIME: {12}" \
                    "\nREPLAY_MEMORY(BUYER/SELLER): {13}/{14}\nEPSILON: {15:4.3f}%".format(
            self.coin_name, self.env_type, episode, MAX_EPISODES, self.current_step, self.steps_left,
            self.balance, self.total_profit, self.hold_coin_quantity,
            BUY_AMOUNT, observation.shape, self.data_datetime[0], self.data_datetime[-1],
            buyer_policy.buyer_memory.size(), seller_policy.seller_memory.size(), epsilon * 100
        )

        print(reset_str)

        self.current_step += 1
        self.steps_left -= 1

        return observation, info_dic

    def step_with_info_dic(self, action, info_dic):
        reward = None
        next_env_status = None

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
            next_info_dic = {
                "change_index": 0.0,
                "coin_krw": -1.0,
                "coin_unit_price": -1.0,
                "coin_quantity": -1.0,
                "commission_fee": -1.0
            }
            if self.args.volume:
                next_observation = np.zeros(shape=(WINDOW_SIZE, SIZE_OF_FEATURE))
            else:
                next_observation = np.zeros(shape=(WINDOW_SIZE, SIZE_OF_FEATURE_WITHOUT_VOLUME))
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

        if not self.args.volume:
            current_x = np.delete(current_x, [2 * (size_idx + 1) for size_idx in range(10)], axis=1)

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

    def get_rl_dataset(self, coin_name, args, train_valid_split=False):
        order_book_class = get_order_book_class(coin_name)
        queryset = naver_order_book_session.query(order_book_class).order_by(order_book_class.base_datetime.asc())
        df = pd.read_sql(queryset.statement, naver_order_book_session.bind)

        # df = df.drop(["id", "base_datetime", "collect_timestamp"], axis=1)
        base_datetime_df = df.filter(["base_datetime"], axis=1)

        df = df.filter(FEATURES, axis=1)

        for feature in FEATURES:
            df[feature].mask(df[feature] == 0.0, 0.1, inplace=True)

        base_datetime_data = pd.to_datetime(base_datetime_df["base_datetime"])
        data = df.values

        dim_0 = data.shape[0] - WINDOW_SIZE + 1
        dim_1 = data.shape[1]

        base_datetime_X = []
        X = np.zeros(shape=(dim_0, WINDOW_SIZE, dim_1))

        for i in range(dim_0):
            X[i] = data[i: i + WINDOW_SIZE]
            base_datetime_X.append(str(base_datetime_data[i + WINDOW_SIZE - 1]))

        base_datetime_X = np.asarray(base_datetime_X)

        total_size = X.shape[0]

        if train_valid_split:
            indices = list(range(total_size))
            train_indices = list(set(indices[:int(total_size * 0.8)]))
            valid_indices = list(set(range(total_size)) - set(train_indices))
            x_train = X[train_indices]
            x_train_base_datetime = base_datetime_X[train_indices]
            x_valid = X[valid_indices]
            x_valid_base_datetime = base_datetime_X[valid_indices]

            train_size = x_train.shape[0]
            valid_size = x_valid.shape[0]

            return x_train, x_train_base_datetime, train_size, x_valid, x_valid_base_datetime, valid_size
        else:
            return X, base_datetime_X, total_size