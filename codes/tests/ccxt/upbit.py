import time

import ccxt

from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt

exchange = ccxt.upbit()

class UpbitOrderBook:
    def __init__(self):
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    def get_ohlcv(self):
        coin_names = self.upbit.get_all_coin_names();

        for idx, coin_name in enumerate(coin_names):
            data = exchange.fetch_ohlcv(coin_name + "/KRW", "1m")
            print("***", idx, coin_name)

            time.sleep(0.1)
            if idx == 0:
                break


if __name__ == "__main__":
    upbit_order_book = UpbitOrderBook()
    upbit_order_book.get_ohlcv()