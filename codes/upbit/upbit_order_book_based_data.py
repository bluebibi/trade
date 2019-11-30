import pickle
import sqlite3

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.utils import get_invest_krw

from common.logger import get_logger

from codes.upbit.upbit_api import Upbit
from common.global_variables import *

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

    def get_original_dataset_for_buy(self):
        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin_recent_window.format(self.coin_name, WINDOW_SIZE),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
        )

        df = df.sort_values(['collect_timestamp', 'base_datetime'], ascending=True)
        return df

    def get_dataset_for_buy(self, model_type="LSTM"):
        df = self.get_original_dataset_for_buy()
        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        if os.path.exists(os.path.join(PROJECT_HOME, "models", "scalers", self.coin_name)):
            try:
                with open(os.path.join(PROJECT_HOME, "models", "scalers", self.coin_name), 'rb') as f:
                    min_max_scaler = pickle.load(f)

                data_normalized = min_max_scaler.transform(df.values)
                data_normalized = torch.from_numpy(data_normalized).float().to(DEVICE)

                if model_type == "LSTM":
                    return data_normalized.unsqueeze(dim=0)
                else:
                    data_normalized = data_normalized.flatten()
                    return data_normalized.unsqueeze(dim=0)
            except Exception:
                return None
        else:
            return None

    def get_original_dataset_for_training(self, limit=False):
        if limit:
            df = pd.read_sql_query(
                select_all_from_order_book_for_one_coin_limit.format(self.coin_name, limit),
                sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
            )
        else:
            #print(select_all_from_order_book_for_one_coin.format(self.coin_name))
            df = pd.read_sql_query(
                select_all_from_order_book_for_one_coin.format(self.coin_name),
                sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
            )
        return df

    def _get_dataset(self, limit=False):
        df = self.get_original_dataset_for_training(limit)
        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        data = torch.from_numpy(df.values).to(DEVICE)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(df.values)
        data_normalized = torch.from_numpy(data_normalized).to(DEVICE)

        with open(os.path.join(PROJECT_HOME, "models", "scalers", self.coin_name), 'wb') as f:
            pickle.dump(min_max_scaler, f)

        x, x_normalized, y, y_up, one_rate, total_size = self.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE
        )

        return x, x_normalized, y, y_up, one_rate, total_size

    def get_rl_dataset(self):
        x, x_normalized, _, _, _, total_size = self._get_dataset(limit=False)

        indices = list(range(total_size))
        train_indices = list(set(indices[:int(total_size * 0.8)]))
        valid_indices = list(set(range(total_size)) - set(train_indices))
        x_train_normalized = x_normalized[train_indices]
        x_valid_normalized = x_normalized[valid_indices]
        train_size = x_train_normalized.size(0)
        valid_size = x_valid_normalized.size(0)

        return x, x_train_normalized, train_size, x_valid_normalized, valid_size

    def get_dataset(self, limit=False, split=True):
        x, x_normalized, y, y_up, one_rate, total_size = self._get_dataset(limit=limit)

        # Imbalanced Preprocessing - Start
        if one_rate > 0.01:
            x_normalized = x_normalized.cpu()
            y_up = y_up.cpu()

            try:
                x_samp, y_up_samp = RandomUnderSampler(sampling_strategy=0.75).fit_sample(
                    x_normalized.reshape((x_normalized.shape[0], x_normalized.shape[1] * x_normalized.shape[2])),
                    y_up
                )
                x_normalized = torch.from_numpy(
                    x_samp.reshape(x_samp.shape[0], x_normalized.shape[1], x_normalized.shape[2])
                ).to(DEVICE)
                y_up = torch.from_numpy(y_up_samp).to(DEVICE)
            except ValueError:
                logger.info("{0} - {1}".format(self.coin_name, "RandomUnderSampler - ValueError"))
                x_normalized = x_normalized.to(DEVICE)
                y_up = y_up.to(DEVICE)
            # Imbalanced Preprocessing - End

            total_size = len(x_normalized)

            if split:
                indices = list(range(total_size))
                np.random.shuffle(indices)

                train_indices = list(set(indices[:int(total_size * 0.8)]))
                validation_indices = list(set(range(total_size)) - set(train_indices))

                x_train_normalized = x_normalized[train_indices]
                x_valid_normalized = x_normalized[validation_indices]

                y_up_train = y_up[train_indices]
                y_up_valid = y_up[validation_indices]

                one_rate_train = y_up_train.sum().float() / y_up_train.size(0)
                one_rate_valid = y_up_valid.sum().float() / y_up_valid.size(0)

                train_size = x_train_normalized.size(0)
                valid_size = x_valid_normalized.size(0)

                return x_train_normalized, y_up_train, one_rate_train, train_size,\
                       x_valid_normalized, y_up_valid, one_rate_valid, valid_size
            else:
                one_rate = y_up.sum().float() / y_up.size(0)
                return x_normalized, y_up, one_rate, total_size
        else:
            return x_normalized, y_up, -1, -1

    @staticmethod
    def build_timeseries(data, data_normalized, window_size, future_target_size, up_rate):
        future_target = future_target_size - 1

        dim_0 = data.shape[0] - window_size - future_target
        dim_1 = data.shape[1]

        x = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)
        x_normalized = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)

        y = torch.zeros((dim_0,)).to(DEVICE)
        y_up = torch.zeros((dim_0,)).float().to(DEVICE)

        for i in range(dim_0):
            x[i] = data[i: i + window_size]
            x_normalized[i] = data_normalized[i: i + window_size]

        count_one = 0
        for i in range(dim_0):
            max_price = -1.0

            ask_price_lst = []
            ask_size_lst = []
            for w in range(0, 60, 4):
                ask_price_lst.append(x[i][-1][1 + w].item())
                ask_size_lst.append(x[i][-1][3 + w].item())

            invest_krw = get_invest_krw(
                current_price=x[i][-1][1].item(),
                total_ask_size=x[i][-1][121],
                total_bid_size=x[i][-1][123]
            )

            original_krw, fee, calc_price, calc_size_sum = upbit.get_expected_buy_coin_price_for_krw_and_ask_list(
                ask_price_lst=ask_price_lst,
                ask_size_lst=ask_size_lst,
                krw=invest_krw,
                transaction_fee_rate=TRANSACTION_FEE_RATE
            )

            for j in range(future_target + 1):
                bid_price_lst = []
                bid_size_lst = []
                for w in range(0, 60, 4):
                    bid_price_lst.append(data[i + window_size + j][61 + w].item())
                    bid_size_lst.append(data[i + window_size + j][63 + w].item())

                original_volume, future_price, fee, future_krw_sum = upbit.get_expected_sell_coin_price_for_volume_and_bid_list(
                    bid_price_lst=bid_price_lst,
                    bid_size_lst=bid_size_lst,
                    volume=calc_size_sum,
                    transaction_fee_rate=TRANSACTION_FEE_RATE
                )

                if future_price > max_price:
                    max_price = future_price

            y[i] = max_price

            if y[i] > calc_price * (1 + up_rate):
                y_up[i] = 1
                count_one += 1

        return x, x_normalized, y, y_up, count_one / dim_0, dim_0


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
    upbit_orderbook_based_data = UpbitOrderBookBasedData("MEDX")

    #upbit_orderbook_based_data.get_data_imbalance_processed()

    x_train_normalized_original, y_up_train_original, one_rate_train, train_size, \
    x_valid_normalized_original, y_up_valid_original, one_rate_valid, valid_size = upbit_orderbook_based_data.get_dataset()

    print(x_train_normalized_original, y_up_train_original, one_rate_train, train_size)


if __name__ == "__main__":
    main()