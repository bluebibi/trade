import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.utils import get_invest_krw

from common.logger import get_logger

from db.sqlite_handler import *
from upbit.upbit_api import Upbit

logger = get_logger("upbit_order_book_based_data")
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


class UpbitOrderBookBasedData:
    def __init__(self, coin_name):
        self.coin_name = coin_name

    def get_dataset_for_buy(self, model_type="LSTM"):
        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin_recent_window.format(self.coin_name, WINDOW_SIZE),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
        )

        df = df.sort_values(['collect_timestamp', 'base_datetime'], ascending=True)

        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        min_max_scaler = MinMaxScaler()
        x_normalized = min_max_scaler.fit_transform(df.values)
        x_normalized = torch.from_numpy(x_normalized).float().to(DEVICE)

        if model_type == "CNN":
            return x_normalized.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            return x_normalized.unsqueeze(dim=0)

    def get_dataset(self):
        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin.format(self.coin_name),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
        )

        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        data = torch.from_numpy(df.values).to(DEVICE)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(df.values)
        data_normalized = torch.from_numpy(data_normalized).to(DEVICE)

        x, x_normalized, y, y_up, one_rate, total_size = self.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE
        )

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
