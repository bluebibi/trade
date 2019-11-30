import os
import sys
import unittest

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.predict.buy import *


class TestBuy(unittest.TestCase):
    def setUp(self):
        pass

    def test_x(self):
        coin_name = "ADA"
        upbit_data = UpbitOrderBookBasedData(coin_name)
        x = upbit_data.get_original_dataset_for_training(limit=False)
        print(x)

        x = upbit_data.get_original_dataset_for_buy()
        print(x)



if __name__ == "__main__":
    unittest.main()
