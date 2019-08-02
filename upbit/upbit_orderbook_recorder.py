import sqlite3
from pytz import timezone
import datetime as dt
import time

import sys, os

from common.utils import convert_to_daily_timestamp

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from common.logger import get_logger
from datetime import timedelta

logger = get_logger("upbit_order_book_recorder_logger")

if os.getcwd().endswith("upbit"):
    os.chdir("..")

order_book_insert_sql = "INSERT INTO KRW_{0}_ORDER_BOOK VALUES(NULL, "\
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?);"

select_by_start_base_datetime = "SELECT base_datetime FROM KRW_{0}_ORDER_BOOK ORDER BY collect_timestamp ASC, base_datetime ASC LIMIT 1;"
select_by_final_base_datetime = "SELECT base_datetime FROM KRW_{0}_ORDER_BOOK ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT 1;"

select_by_datetime = "SELECT base_datetime FROM KRW_{0}_ORDER_BOOK WHERE base_datetime=? LIMIT 1;"

class UpbitOrderBookRecorder:
    def __init__(self):
        self.coin_names = UPBIT.get_all_coin_names()

    def get_order_book_start_and_final(self, coin_name):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None,
                             check_same_thread=False) as conn:
            cursor = conn.cursor()

            cursor.execute(select_by_start_base_datetime.format(coin_name))
            start_base_datetime_str = cursor.fetchone()[0]

            cursor.execute(select_by_final_base_datetime.format(coin_name))
            final_base_datetime_str = cursor.fetchone()[0]

            conn.commit()
        return start_base_datetime_str, final_base_datetime_str

    def get_order_book_consecutiveness(self, coin_name=None, start_base_datetime_str=None):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None,
                             check_same_thread=False) as conn:
            cursor = conn.cursor()

            if start_base_datetime_str is None:
                cursor.execute(select_by_start_base_datetime.format(coin_name))
                start_base_datetime_str = cursor.fetchone()[0]

            cursor.execute(select_by_datetime.format(coin_name), (start_base_datetime_str,))
            start_base_datetime_str = cursor.fetchone()

            if start_base_datetime_str is None:
                return None

            base_datetime = dt.datetime.strptime(start_base_datetime_str[0], fmt.replace("T", " "))

            last_base_datetime_str = None
            while True:
                next_base_datetime = base_datetime + timedelta(minutes=1)
                next_base_datetime_str = dt.datetime.strftime(next_base_datetime, fmt.replace("T", " "))
                cursor.execute(select_by_datetime.format(coin_name), (next_base_datetime_str, ))
                next_base_datetime_str = cursor.fetchone()

                if not next_base_datetime_str:
                    break

                next_base_datetime_str = next_base_datetime_str[0]
                last_base_datetime_str = next_base_datetime_str
                base_datetime = dt.datetime.strptime(next_base_datetime_str, fmt.replace("T", " "))

            conn.commit()

        return last_base_datetime_str


    def record(self, base_datetime, coin_ticker_name):
        daily_base_timestamp = convert_to_daily_timestamp(base_datetime)

        order_book = UPBIT.get_orderbook(tickers=coin_ticker_name)[0]

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

        collect_timestamp = order_book['timestamp']
        total_ask_size = order_book['total_ask_size']
        total_bid_size = order_book['total_bid_size']

        return {"base_datetime": base_datetime,
                "daily_base_timestamp": daily_base_timestamp,
                "collect_timestamp": collect_timestamp,
                "ask_price_lst": ask_price_lst,
                "ask_size_lst": ask_size_lst,
                "bid_price_lst": bid_price_lst,
                "bid_size_lst": bid_size_lst,
                "total_ask_size": total_ask_size,
                "total_bid_size": total_bid_size}

    def insert_order_book(self, order_book_info):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None,
                             check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in order_book_info:
                cursor.execute(order_book_insert_sql.format(coin_name), (
                    order_book_info[coin_name]["base_datetime"],
                    int(order_book_info[coin_name]["daily_base_timestamp"]),
                    int(order_book_info[coin_name]["collect_timestamp"]),

                    float(order_book_info[coin_name]["ask_price_lst"][0]),
                    float(order_book_info[coin_name]["ask_size_lst"][0]),

                    float(order_book_info[coin_name]["ask_price_lst"][1]),
                    float(order_book_info[coin_name]["ask_size_lst"][1]),

                    float(order_book_info[coin_name]["ask_price_lst"][2]),
                    float(order_book_info[coin_name]["ask_size_lst"][2]),

                    float(order_book_info[coin_name]["ask_price_lst"][3]),
                    float(order_book_info[coin_name]["ask_size_lst"][3]),

                    float(order_book_info[coin_name]["ask_price_lst"][4]),
                    float(order_book_info[coin_name]["ask_size_lst"][4]),

                    float(order_book_info[coin_name]["ask_price_lst"][5]),
                    float(order_book_info[coin_name]["ask_size_lst"][5]),

                    float(order_book_info[coin_name]["ask_price_lst"][6]),
                    float(order_book_info[coin_name]["ask_size_lst"][6]),

                    float(order_book_info[coin_name]["ask_price_lst"][7]),
                    float(order_book_info[coin_name]["ask_size_lst"][7]),

                    float(order_book_info[coin_name]["ask_price_lst"][8]),
                    float(order_book_info[coin_name]["ask_size_lst"][8]),

                    float(order_book_info[coin_name]["ask_price_lst"][9]),
                    float(order_book_info[coin_name]["ask_size_lst"][9]),

                    float(order_book_info[coin_name]["ask_price_lst"][10]),
                    float(order_book_info[coin_name]["ask_size_lst"][10]),

                    float(order_book_info[coin_name]["ask_price_lst"][11]),
                    float(order_book_info[coin_name]["ask_size_lst"][11]),

                    float(order_book_info[coin_name]["ask_price_lst"][12]),
                    float(order_book_info[coin_name]["ask_size_lst"][12]),

                    float(order_book_info[coin_name]["ask_price_lst"][13]),
                    float(order_book_info[coin_name]["ask_size_lst"][13]),

                    float(order_book_info[coin_name]["ask_price_lst"][14]),
                    float(order_book_info[coin_name]["ask_size_lst"][14]),

                    float(order_book_info[coin_name]["bid_price_lst"][0]),
                    float(order_book_info[coin_name]["bid_size_lst"][0]),

                    float(order_book_info[coin_name]["bid_price_lst"][1]),
                    float(order_book_info[coin_name]["bid_size_lst"][1]),

                    float(order_book_info[coin_name]["bid_price_lst"][2]),
                    float(order_book_info[coin_name]["bid_size_lst"][2]),

                    float(order_book_info[coin_name]["bid_price_lst"][3]),
                    float(order_book_info[coin_name]["bid_size_lst"][3]),

                    float(order_book_info[coin_name]["bid_price_lst"][4]),
                    float(order_book_info[coin_name]["bid_size_lst"][4]),

                    float(order_book_info[coin_name]["bid_price_lst"][5]),
                    float(order_book_info[coin_name]["bid_size_lst"][5]),

                    float(order_book_info[coin_name]["bid_price_lst"][6]),
                    float(order_book_info[coin_name]["bid_size_lst"][6]),

                    float(order_book_info[coin_name]["bid_price_lst"][7]),
                    float(order_book_info[coin_name]["bid_size_lst"][7]),

                    float(order_book_info[coin_name]["bid_price_lst"][8]),
                    float(order_book_info[coin_name]["bid_size_lst"][8]),

                    float(order_book_info[coin_name]["bid_price_lst"][9]),
                    float(order_book_info[coin_name]["bid_size_lst"][9]),

                    float(order_book_info[coin_name]["bid_price_lst"][10]),
                    float(order_book_info[coin_name]["bid_size_lst"][10]),

                    float(order_book_info[coin_name]["bid_price_lst"][11]),
                    float(order_book_info[coin_name]["bid_size_lst"][11]),

                    float(order_book_info[coin_name]["bid_price_lst"][12]),
                    float(order_book_info[coin_name]["bid_size_lst"][12]),

                    float(order_book_info[coin_name]["bid_price_lst"][13]),
                    float(order_book_info[coin_name]["bid_size_lst"][13]),

                    float(order_book_info[coin_name]["bid_price_lst"][14]),
                    float(order_book_info[coin_name]["bid_size_lst"][14]),

                    float(order_book_info[coin_name]["total_ask_size"]),
                    float(order_book_info[coin_name]["total_bid_size"])
                ))
            conn.commit()


if __name__ == "__main__":
    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    base_datetime = current_time_str[:-3] + ":00"

    upbit_order_book_recorder = UpbitOrderBookRecorder()

    current_time = time.time()

    order_book_info = {}
    for coin_name in upbit_order_book_recorder.coin_names:
        order_book_info[coin_name] = upbit_order_book_recorder.record(
            base_datetime=base_datetime,
            coin_ticker_name="KRW-" + coin_name
        )
        time.sleep(0.2)

    upbit_order_book_recorder.insert_order_book(order_book_info)

    elapsed_time = time.time() - current_time
    logger.info("{0} - Elapsed Time: {1} - Num of coins: {2}".format(base_datetime, elapsed_time, len(order_book_info)))
