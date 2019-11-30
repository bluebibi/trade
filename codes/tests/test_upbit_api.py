import sqlite3
import unittest
from pytz import timezone

import sys, os

from sklearn.preprocessing import MinMaxScaler

from common.utils import convert_to_daily_timestamp, get_invest_krw
from codes.upbit.upbit_order_book_based_data import UpbitOrderBookBasedData

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from codes.upbit.upbit_api import Upbit
import pprint
import datetime as dt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from common.logger import get_logger

logger = get_logger("test_upbit")

from codes.upbit.upbit_order_book_recorder import UpbitOrderBookRecorder

pp = pprint.PrettyPrinter(indent=2)


class UpBitAPITestCase(unittest.TestCase):
    def setUp(self):
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    def test_get_tickers(self):
        pp.pprint(self.upbit.get_tickers(fiat="KRW"))

    def test_get_ohlcv(self):
        # print(get_ohlcv("KRW-BTC"))
        # print(get_ohlcv("KRW-BTC", interval="day", count=5))
        # print(get_ohlcv("KRW-BTC", interval="minute1"))
        # print(get_ohlcv("KRW-BTC", interval="minute3"))
        # print(get_ohlcv("KRW-BTC", interval="minute5"))
        pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute10"))
        # print(get_ohlcv("KRW-BTC", interval="minute15"))
        # pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute30"))
        # print(get_ohlcv("KRW-BTC", interval="minute60"))
        # print(get_ohlcv("KRW-BTC", interval="minute240"))
        # print(get_ohlcv("KRW-BTC", interval="week"))
        # pp.pprint(self.upbit.get_daily_ohlcv_from_base("KRW-BTC", base=9))
        # print(get_ohlcv("KRW-BTC", interval="day", count=5))

    def test_get_current_price(self):
        # print(get_current_price("KRW-BTC"))
        pp.pprint(self.upbit.get_current_price(
            ['KRW-GAS', 'KRW-MOC', 'KRW-IQ', 'KRW-WAX', 'KRW-NEO', 'KRW-AERGO', 'KRW-MEDX', 'KRW-XMR',
             'KRW-OST', 'KRW-STRAT', 'KRW-IOST', 'KRW-ONT', 'KRW-BSV']))

    def test_get_right_invest_krw(self):
        coin_names = self.upbit.get_all_coin_names();

        for coin_name in coin_names:

            info = self.upbit.get_orderbook(tickers="KRW-"+coin_name)
            current_price = info[0]['orderbook_units'][0]['ask_price']
            total_ask_size = info[0]['total_ask_size']
            total_bid_size = info[0]['total_bid_size']
            print(info)
            print(current_price, total_ask_size, total_bid_size)
            print("{0}, {1:9.0f}".format(coin_name, get_invest_krw(current_price, total_ask_size=total_ask_size,
                                                                   total_bid_size=total_bid_size)))





    #### 매우 중요 <-- Missing Data 처리
    def test_order_book_consecutiveness(self):
        print()
        coin_names = self.upbit.get_all_coin_names();

        order_book_insert_sql = "INSERT INTO KRW_{0}_ORDER_BOOK VALUES(NULL, " \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                                "?, ?, ?, ?, ?);"

        select_by_datetime = "SELECT * FROM KRW_{0}_ORDER_BOOK WHERE base_datetime=? LIMIT 1;"

        upbit_order_book_recorder = UpbitOrderBookRecorder()


        for coin_name in coin_names:
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


                    ### SWITCH
                    #self.insert_missing_record(select_by_datetime, coin_name, previous_base_datetime_str, order_book_insert_sql, start_base_datetime_str)

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

            print()

    def insert_missing_record(self, select_by_datetime, coin_name, previous_base_datetime_str, order_book_insert_sql, start_base_datetime_str):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10,
                             check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(select_by_datetime.format(coin_name), (previous_base_datetime_str,))
            info = cursor.fetchone()

            logger.info(info)

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

    def test_bulk_order_book(self):
        coin_names = self.upbit.get_all_coin_names()
        for i, coin_name in enumerate(coin_names):
            order_book = self.upbit.get_orderbook("KRW-" + coin_name)
            print(i, order_book[0]['market'])

    def test_get_order_book(self):
        now = dt.datetime.now(timezone('Asia/Seoul'))
        now_str = now.strftime(fmt)
        current_time_str = now_str.replace("T", " ")
        current_time_str = current_time_str[:-3] + ":00"

        order_book = self.upbit.get_orderbook(tickers="KRW-BTC")[0]

        order_book_units = order_book["orderbook_units"]
        ask_price_lst = []
        ask_size_lst = []
        bid_price_lst = []
        bid_size_lst = []
        for item in order_book_units:
            ask_price_lst.append(item["ask_price"])
            ask_size_lst.append(item["ask_size"])
            bid_price_lst.append(item["bid_price"])
            bid_size_lst.append(item["bid_size"])

        timestamp = order_book['timestamp']
        total_ask_size = order_book['total_ask_size']
        total_bid_size = order_book['total_bid_size']


        logger.info(current_time_str)
        logger.info(order_book)

    def test_get_market_index(self):
        pp.pprint(self.upbit.get_market_index())

    def test_get_all_coin_names(self):
        coin_names = self.upbit.get_all_coin_names()
        print(coin_names)
        print(len(coin_names))

    def test_get_coin_info(self):
        pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute10"))
        pp.pprint(self.upbit.get_orderbook(tickers="KRW-BTC"))
        pp.pprint(self.upbit.get_market_index())
        pass

    def test_get_expected_buy_coin_price_for_krw(self):
        expected_price = self.upbit.get_expected_buy_coin_price_for_krw("KRW-OMG", 1000000, TRANSACTION_FEE_RATE)
        print(expected_price)

    def test_get_expected_sell_coin_price_for_volume(self):
        expected_price = self.upbit.get_expected_sell_coin_price_for_volume("KRW-OMG", 548.7964360338357, TRANSACTION_FEE_RATE)
        print(expected_price)

    def test_get_balance(self):
        pp.pprint(self.upbit.get_balances())

        # 원화 잔고 조회
        print(self.upbit.get_balance(ticker="KRW"))
        print(self.upbit.get_balance(ticker="KRW-BTC"))
        print(self.upbit.get_balance(ticker="KRW-XRP"))

        # 매도
        # print(upbit.sell_limit_order("KRW-XRP", 1000, 20))

        # 매수
        # print(upbit.buy_limit_order("KRW-XRP", 200, 20))

        # 주문 취소
        # print(upbit.cancel_order('82e211da-21f6-4355-9d76-83e7248e2c0c'))

    def test_get_data_imbalance_processed(self):
        coin_name = "BTC"
        order_book_based_data = UpbitOrderBookBasedData(coin_name)

        df = pd.read_sql_query(
            select_all_from_order_book_for_one_coin.format(coin_name),
            sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False)
        )

        df = df.drop(["base_datetime", "collect_timestamp"], axis=1)

        data = torch.from_numpy(df.values).to(DEVICE)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(df.values)
        data_normalized = torch.from_numpy(data_normalized).to(DEVICE)

        x, x_normalized, y, y_up, one_rate, total_size = order_book_based_data.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE,
            scaler=min_max_scaler
        )

        print()
        print(x_normalized.shape, y_up.shape)
        print("one_rate: {0}, total_size: {1}".format(one_rate, total_size))

        print()

        x_samp, y_up_samp = RandomUnderSampler(sampling_strategy=0.75).fit_sample(
            x_normalized.reshape((x_normalized.shape[0], x_normalized.shape[1] * x_normalized.shape[2])),
            y_up
        )
        one_rate = (y_up_samp == 1).sum() / len(y_up_samp)

        x_samp = torch.from_numpy(x_samp.reshape(x_samp.shape[0], x_normalized.shape[1], x_normalized.shape[2]))
        y_up_samp = torch.from_numpy(y_up_samp)

        print(x_samp.shape, y_up_samp.shape)
        print("one_rate: {0}, total_size: {1}".format(one_rate, len(x_samp)))

if __name__ == '__main__':
    unittest.main()