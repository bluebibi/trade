import os
import sys
import unittest

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from predict.buy import *
from common.global_variables import SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2


class TestSlack(unittest.TestCase):
    def setUp(self):
        pass

    def test_send_message(self):
        q = get_db_right_time_coin_names()
        print(q)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
