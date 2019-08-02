from common.global_variables import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import sqlite3
import datetime as dt

from common.utils import convert_to_daily_timestamp
from upbit.upbit_orderbook_recorder import UpbitOrderBookRecorder

from common.logger import get_logger

logger = get_logger("upbit_orderbook_based_data")

order_book_insert_sql = """
    INSERT INTO KRW_{0}_ORDER_BOOK VALUES(
    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ?, ?, ?, ?, ?);
"""

select_order_book_by_datetime = """
    SELECT * FROM KRW_{0}_ORDER_BOOK WHERE base_datetime=? LIMIT 1;
"""

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

select_all_from_order_book_for_one_coin = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp ASC, base_datetime ASC;
"""

select_all_from_order_book_for_one_coin_recent_window = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT {1};
"""

#print(select_all_from_order_book_for_one_coin)


class UpbitOrderBookBasedData:
    def __init__(self, coin_name):
        self.coin_name = coin_name
        self.processing_missing_data("BTC")
        self.processing_missing_data(self.coin_name)

    def processing_missing_data(self, coin_name):
        logger.info("Processing Missing Data")

        upbit_order_book_recorder = UpbitOrderBookRecorder()

        start_base_datetime_str, final_base_datetime_str = upbit_order_book_recorder.get_order_book_start_and_final(coin_name)
        logger.info("{0:5s} - Start: {1}, Final: {2}".format(
            coin_name,
            start_base_datetime_str,
            final_base_datetime_str
        ))

        missing_count = 0
        while True:
            last_base_datetime_str = upbit_order_book_recorder.get_order_book_consecutiveness(
                coin_name=coin_name,
                start_base_datetime_str=start_base_datetime_str
            )

            if last_base_datetime_str == final_base_datetime_str:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                break

            if last_base_datetime_str is None:
                missing_count += 1
                logger.info("{0:5s} - Start Base Datetime: {1} - Missing: {2}".format(
                    coin_name, start_base_datetime_str, missing_count
                ))
                previous_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                previous_base_datetime = previous_base_datetime - dt.timedelta(minutes=1)
                previous_base_datetime_str = dt.datetime.strftime(previous_base_datetime, fmt.replace("T", " "))

                self.insert_missing_record(
                    coin_name,
                    previous_base_datetime_str,
                    order_book_insert_sql,
                    start_base_datetime_str
                )

                start_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=1)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))
            else:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                start_base_datetime = dt.datetime.strptime(last_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=1)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))

    def insert_missing_record(self, coin_name, previous_base_datetime_str, order_book_insert_sql, start_base_datetime_str):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(select_order_book_by_datetime.format(coin_name), (previous_base_datetime_str,))
            info = cursor.fetchone()

            cursor.execute(order_book_insert_sql.format(coin_name), (
                start_base_datetime_str, convert_to_daily_timestamp(start_base_datetime_str), info[3],
                info[4], info[5], info[6], info[7], info[8], info[9], info[10], info[11], info[12],
                info[13],
                info[14], info[15], info[16], info[17], info[18], info[19], info[20], info[21], info[22],
                info[23],
                info[24], info[25], info[26], info[27], info[28], info[29], info[30], info[31], info[32],
                info[33],
                info[34], info[35], info[36], info[37], info[38], info[39], info[40], info[41], info[42],
                info[43],
                info[44], info[45], info[46], info[47], info[48], info[49], info[50], info[51], info[52],
                info[53],
                info[54], info[55], info[56], info[57], info[58], info[59], info[60], info[61], info[62],
                info[63],
                info[64], info[65]
            ))
            conn.commit()

    def get_buy_for_data(self, model_type):
        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin_recent_window.format(self.coin_name, WINDOW_SIZE),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None, check_same_thread=False)
        )

        df = df.sort_values(['collect_timestamp', 'base_datetime'], ascending=True)

        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        #print(df)

        min_max_scaler = MinMaxScaler()
        x_normalized = min_max_scaler.fit_transform(df.values)
        x_normalized = torch.from_numpy(x_normalized).float().to(DEVICE)

        if model_type == "CNN":
            return x_normalized.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            return x_normalized.unsqueeze(dim=0)

    def get_data(self, model_type):
        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin.format(self.coin_name),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None, check_same_thread=False)
        )

        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        #print(df)

        data = torch.from_numpy(df.values).to(DEVICE)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(df.values)
        data_normalized = torch.from_numpy(data_normalized).to(DEVICE)

        x, x_normalized, y, y_normalized, y_up, one_rate, total_size = self.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE
        )

        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = list(set(indices[:int(total_size * 0.8)]))
        validation_indices = list(set(range(total_size)) - set(train_indices))

        x_train = x[train_indices]
        x_train_normalized = x_normalized[train_indices]

        x_valid = x[validation_indices]
        x_valid_normalized = x_normalized[validation_indices]

        y_train = y[train_indices]
        y_train_normalized = y_normalized[train_indices]

        y_valid = y[validation_indices]
        y_valid_normalized = y_normalized[validation_indices]

        y_up_train = y_up[train_indices]
        y_up_valid = y_up[validation_indices]

        one_rate_train = y_up_train.sum().float() / y_up_train.size(0)
        one_rate_valid = y_up_valid.sum().float() / y_up_valid.size(0)

        train_size = x_train.size(0)
        valid_size = x_valid.size(0)

        if model_type == "CNN":
            return x_train.unsqueeze(dim=1), x_train_normalized.unsqueeze(dim=1), y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid.unsqueeze(dim=1), x_valid_normalized.unsqueeze(dim=1), y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size
        else:
            return x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size

    @staticmethod
    def build_timeseries(data, data_normalized, window_size, future_target_size, up_rate):
        y_col_index = 3
        future_target = future_target_size - 1

        dim_0 = data.shape[0] - window_size - future_target
        dim_1 = data.shape[1]

        x = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)
        x_normalized = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)

        y = torch.zeros((dim_0,)).to(DEVICE)
        y_normalized = torch.zeros((dim_0,)).to(DEVICE)
        y_up = torch.zeros((dim_0,)).float().to(DEVICE)

        for i in range(dim_0):
            x[i] = data[i: i + window_size]
            x_normalized[i] = data_normalized[i: i + window_size]

        count_one = 0
        for i in range(dim_0):
            max_price = -1.0
            max_price_normalized = -1.0

            for j in range(future_target + 1):
                future_price = data[i + window_size + j, y_col_index]
                future_price_normalized = data_normalized[i + window_size + j, y_col_index]

                if future_price > max_price:
                    max_price = future_price
                    max_price_normalized = future_price_normalized

            y[i] = max_price
            y_normalized[i] = max_price_normalized

            if y[i] > x[i][-1, y_col_index] * (1 + up_rate):
                y_up[i] = 1
                count_one += 1

        return x, x_normalized, y, y_normalized, y_up, count_one / dim_0, dim_0


def get_data_loader(x, x_normalized, y, y_normalized, y_up_train, batch_size, suffle=True):
    total_size = x.size(0)
    if total_size % batch_size == 0:
        num_batches = int(total_size / batch_size)
    else:
        num_batches = int(total_size / batch_size) + 1

    for i in range(num_batches):
        if suffle:
            indices = np.random.choice(total_size, batch_size)
        else:
            indices = np.asarray(range(i * batch_size, min((i + 1) * batch_size, total_size)))

        yield x[indices], x_normalized[indices], y[indices], y_normalized[indices], y_up_train[indices], num_batches


def main():
    upbit_orderbook_based_data = UpbitOrderBookBasedData("ADA")

    upbit_orderbook_based_data.get_data("CNN")

    upbit_orderbook_based_data.get_buy_for_data("CNN")


if __name__ == "__main__":
    main()
