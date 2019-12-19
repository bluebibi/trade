import os
import sys
import unittest

import pandas as pd

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.predict.buy import *


class TestBuy(unittest.TestCase):
    def setUp(self):
        pass

    # def test_x(self):
    #     coin_name = "ADA"
    #     upbit_data = UpbitOrderBookBasedData(coin_name)
    #     df = pd.read_sql_table(upbit_data.order_book_class.__tablename__, order_book_conn)
    #     df = df.drop(["id", "base_datetime", "collect_timestamp"], axis=1)
    #     data = df.values
    #     # x = upbit_data.get_original_dataset_for_buy()
    #     print(data.shape)
    #     print(data[0:10])

    def test_get_db_right_time_coin_names(self):
        l = get_db_right_time_coin_names()
        print(l)



if __name__ == "__main__":
    unittest.main()
