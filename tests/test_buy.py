import os
import sys
import unittest

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from predict.buy import *
from common.global_variables import SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2
from db.sqlite_handler import *

class TestBuy(unittest.TestCase):
    def setUp(self):
        pass

    def test_base_timestamp(self):
        current_time_str = "2019-08-04 15:28:00"

        print(current_time_str)

        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            all_coin_names = upbit.get_all_coin_names()
            for coin_name in all_coin_names:
                cursor.execute(select_current_base_datetime_by_datetime.format(coin_name, current_time_str))
                base_datetime = cursor.fetchone()
                if base_datetime is not None:
                    print(base_datetime)

    def test_buy(self):
        buyer = UpbitOrderBookBasedData("BTC")
        d = buyer.get_buy_for_data("CNN")
        print(d)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
