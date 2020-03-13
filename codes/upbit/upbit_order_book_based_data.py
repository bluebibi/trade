import gc
import pickle
import sqlite3

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import sys, os

from sklearn.utils import shuffle

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.utils import get_invest_krw
from common.logger import get_logger
from codes.upbit.upbit_api import Upbit
from common.global_variables import *
from web.db.database import get_order_book_class, naver_order_book_session


logger = get_logger("upbit_order_book_based_data")

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

order_book_for_one_coin = """
    SELECT 
    C.base_datetime as 'base_datetime', 
    C.daily_base_timestamp as 'daily_base_timestamp', 
    C.collect_timestamp as 'collect_timestamp',
    
    {0}
"""

sql_body = ""

one_sql_ask_body = """
    C.ask_price_{0} as 'ask_price_{0}',
    B.ask_price_{0} as 'ask_price_{0}_btc',
    C.ask_size_{0} as 'ask_size_{0}',
    B.ask_size_{0} as 'ask_size_{0}_btc',
"""

one_sql_bid_body = """
    C.bid_price_{0} as 'bid_price_{0}',
    B.bid_price_{0} as 'bid_price_{0}_btc',
    C.bid_size_{0} as 'bid_size_{0}',
    B.bid_size_{0} as 'bid_size_{0}_btc',
"""

for i in range(0, 15):
    sql_body += one_sql_ask_body.format(i)

for i in range(0, 15):
    sql_body += one_sql_bid_body.format(i)

sql_body += """
    C.total_ask_size as 'total_ask_size', 
    B.total_ask_size as 'total_ask_size_btc',
    C.total_bid_size as 'total_bid_size', 
    B.total_bid_size as 'total_bid_size_btc'    
    """

order_book_for_one_coin = order_book_for_one_coin.format(sql_body)

select_all_from_order_book_for_one_coin_recent_window = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT {1};
"""

select_all_from_order_book_for_one_coin = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp ASC, base_datetime ASC;
"""

select_all_from_order_book_for_one_coin_limit = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp ASC, base_datetime ASC LIMIT {1};
"""


class UpbitOrderBookBasedData:
    def __init__(self, coin_name):
        self.coin_name = coin_name
        self.order_book_class = get_order_book_class(coin_name)

    def get_dataset_for_buy(self, model_type="GB"):
        queryset = naver_order_book_session.query(self.order_book_class).order_by(
            self.order_book_class.base_datetime.desc()
        ).limit(WINDOW_SIZE)

        df = pd.read_sql(queryset.statement, naver_order_book_session.bind)
        df = df.sort_values(['base_datetime'], ascending=True)
        df = df.drop(["id", "base_datetime", "collect_timestamp"], axis=1)

        if os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, "SCALERS", self.coin_name)):
            with open(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, "SCALERS", self.coin_name), 'rb') as f:
                min_max_scaler = pickle.load(f)

            data_normalized = min_max_scaler.transform(df.values)

            if model_type != "LSTM":
                data_normalized = data_normalized.flatten()

            return np.expand_dims(data_normalized, axis=0)
        else:
            return None

    def _get_dataset(self, for_rl=False):
        queryset = naver_order_book_session.query(self.order_book_class).order_by(self.order_book_class.base_datetime.asc())
        df = pd.read_sql(queryset.statement, naver_order_book_session.bind)
        #df = df.drop(["id", "base_datetime", "collect_timestamp"], axis=1)
        df = df.filter(["ask_price_0", "ask_size_0", "ask_price_1", "ask_size_1", "ask_price_2", "ask_size_2", "bid_price_0", "bid_size_0", "bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2"], axis=1)
        if for_rl:
            X = self.build_timeseries_for_rl(data=df.values)
            return X
        else:
            X, y_up, one_rate, total_size = self.build_timeseries(data=df.values)
            return X, y_up, one_rate, total_size

    def get_rl_dataset(self):
        x_normalized = self._get_dataset(for_rl=True)

        total_size = x_normalized.shape[0]

        indices = list(range(total_size))
        train_indices = list(set(indices[:int(total_size * 0.8)]))
        valid_indices = list(set(range(total_size)) - set(train_indices))
        x_train_normalized = x_normalized[train_indices]
        x_valid_normalized = x_normalized[valid_indices]

        train_size = x_train_normalized.shape[0]
        valid_size = x_valid_normalized.shape[0]

        return x_train_normalized, train_size, x_valid_normalized, valid_size

    def get_dataset(self, split=True):
        gc.collect()
        X, y_up, one_rate, total_size = self._get_dataset()

        window_size = X.shape[1]
        input_size = X.shape[2]

        # Imbalanced Preprocessing - Start
        if one_rate < 0.25:
            try:
                X_samples, y_up_samples = RandomUnderSampler(sampling_strategy=0.75).fit_sample(
                    X.reshape((X.shape[0], window_size * input_size)),
                    y_up
                )
                X_samples, y_up_samples = shuffle(X_samples, y_up_samples)

                X = X_samples
                y_up = y_up_samples

                total_size = len(X_samples)
                one_rate = sum(y_up) / total_size
            except ValueError:
                logger.info("{0} - {1}".format(self.coin_name, "RandomUnderSampler - ValueError"))
        # Imbalanced Preprocessing - End

        min_max_scaler = MinMaxScaler()

        if split:
            indices = list(range(total_size))
            np.random.shuffle(indices)

            train_indices = list(set(indices[:int(total_size * 0.8)]))
            test_indices = list(set(range(total_size)) - set(train_indices))

            X_train = X[train_indices]
            X_test = X[test_indices]

            y_up_train = y_up[train_indices]
            y_up_test = y_up[test_indices]

            one_rate_train = sum(y_up_train) / y_up_train.shape[0]
            one_rate_test = sum(y_up_test) / y_up_test.shape[0]

            train_size = X_train.shape[0]
            test_size = X_test.shape[0]

            X_train_normalized = min_max_scaler.fit_transform(X_train)
            X_test_normalized = min_max_scaler.transform(X_test)

            with open(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, "SCALERS", self.coin_name), 'wb') as f:
                pickle.dump(min_max_scaler, f)

            X_train_normalized = np.reshape(X_train_normalized, newshape=(
                X_train_normalized.shape[0], window_size, input_size
            ))
            X_test_normalized = np.reshape(X_test_normalized, newshape=(
                X_test_normalized.shape[0], window_size, input_size
            ))
            return X_train_normalized, y_up_train, one_rate_train, train_size, \
                   X_test_normalized, y_up_test, one_rate_test, test_size
        else:
            X_normalized = min_max_scaler.fit_transform(X)

            X_normalized = np.reshape(X_normalized, newshape=(
                X_normalized.shape[0], window_size, input_size
            ))

            with open(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, "SCALERS", self.coin_name), 'wb') as f:
                pickle.dump(min_max_scaler, f)

            return X_normalized, y_up, one_rate, total_size


    @staticmethod
    def build_timeseries_for_rl(data):

        dim_0 = data.shape[0] - WINDOW_SIZE
        dim_1 = data.shape[1]

        X = np.zeros(shape=(dim_0, WINDOW_SIZE, dim_1))

        for i in range(dim_0):
            X[i] = data[i: i + WINDOW_SIZE]

        for i in range(dim_0):
            X[i] = X[i] / X[i][0]

        return X

    @staticmethod
    def build_timeseries(data):
        future_target = FUTURE_TARGET_SIZE - 1

        dim_0 = data.shape[0] - WINDOW_SIZE - future_target
        dim_1 = data.shape[1]

        X = np.zeros(shape=(dim_0, WINDOW_SIZE, dim_1))

        y = np.zeros(dim_0,)
        y_up = np.zeros(dim_0,)

        for i in range(dim_0):
            X[i] = data[i: i + WINDOW_SIZE]

        count_one = 0
        for i in range(dim_0):
            ask_price_lst = []
            ask_size_lst = []
            for w in range(0, 30, 2):
                ask_price_lst.append(X[i][-1][1 + w].item())
                ask_size_lst.append(X[i][-1][2 + w].item())

            invest_krw = get_invest_krw()

            original_krw, fee, calc_price, calc_size_sum = upbit.get_expected_buy_coin_price_for_krw_and_ask_list(
                ask_price_lst=ask_price_lst,
                ask_size_lst=ask_size_lst,
                krw=invest_krw,
                transaction_fee_rate=TRANSACTION_FEE_RATE
            )

            max_price = -1.0
            for j in range(FUTURE_TARGET_SIZE):
                bid_price_lst = []
                bid_size_lst = []
                for w in range(0, 30, 2):
                    bid_price_lst.append(data[i + WINDOW_SIZE + j][31 + w])
                    bid_size_lst.append(data[i + WINDOW_SIZE + j][32 + w])

                original_volume, future_price, fee, future_krw_sum = upbit.get_expected_sell_coin_price_for_volume_and_bid_list(
                    bid_price_lst=bid_price_lst,
                    bid_size_lst=bid_size_lst,
                    volume=calc_size_sum,
                    transaction_fee_rate=TRANSACTION_FEE_RATE
                )

                if future_price > max_price:
                    max_price = future_price

            y[i] = max_price

            if y[i] > calc_price * (1 + UP_RATE):
                y_up[i] = 1
                count_one += 1

        return X, y_up, count_one / dim_0, dim_0


def get_data_loader(x_normalized, y_up, batch_size, shuffle=True):
    total_size = x_normalized.size(0)
    if total_size % batch_size == 0:
        num_batches = int(total_size / batch_size)
    else:
        num_batches = int(total_size / batch_size) + 1

    for idx in range(num_batches):
        if shuffle:
            indices = np.random.choice(total_size, batch_size)
        else:
            indices = np.asarray(range(idx * batch_size, min((idx + 1) * batch_size, total_size)))

        yield x_normalized[indices], y_up[indices], num_batches


def main():
    upbit_orderbook_based_data = UpbitOrderBookBasedData("NPXS")
    x_train_normalized, train_size, x_valid_normalized, valid_size = upbit_orderbook_based_data.get_rl_dataset()

    print(x_train_normalized.shape, train_size, x_valid_normalized.shape, valid_size)

    #upbit_orderbook_based_data.get_data_imbalance_processed()

    # x_train_normalized_original, y_up_train_original, one_rate_train, train_size, \
    # x_valid_normalized_original, y_up_valid_original, one_rate_valid, valid_size = upbit_orderbook_based_data.get_dataset()
    #
    # print(x_train_normalized_original, y_up_train_original, one_rate_train, train_size)


if __name__ == "__main__":
    main()
