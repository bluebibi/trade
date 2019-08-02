import time
import sqlite3
from common.global_variables import *
from common.logger import get_logger

logger = get_logger("upbit_recorder_logger")

if os.getcwd().endswith("upbit"):
    os.chdir("..")

price_insert = "INSERT INTO {0} VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
select_by_datetime = "SELECT * FROM {0} WHERE datetime='{1}';"
remove_duplicated_datetime_sql = """
DELETE FROM {0} WHERE id in
(SELECT id FROM
(SELECT id, "datetime", COUNT(*) c FROM {0} GROUP BY "datetime" HAVING c > 1))
"""


class UpbitRecorder:
    def __init__(self):
        self.coin_names = UPBIT.get_all_coin_names()

    def record(self, coin_name):
        i = UPBIT.get_market_index()

        time.sleep(0.01)

        ticker = "KRW-" + coin_name
        r = UPBIT.get_ohlcv(ticker, interval="minute10").values

        new_records = 0

        for row in r:
            datetime = row[0].replace('T', ' ')
            open_price = row[1]
            high_price = row[2]
            low_price = row[3]
            close_price = row[4]
            volume = row[5]

            if not self.exist_row_by_datetime(coin_name, datetime):
                o = UPBIT.get_orderbook(tickers=ticker)

                with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
                    cursor = conn.cursor()

                    cursor.execute(price_insert.format("KRW_" + coin_name), (
                        datetime,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        o[0]['total_ask_size'],
                        o[0]['total_bid_size'],
                        i['data']['btmi']['market_index'],
                        i['data']['btmi']['rate'],
                        i['data']['btai']['market_index'],
                        i['data']['btai']['rate']
                    ))
                    conn.commit()

                    new_records += 1

                    time.sleep(0.01)

        return new_records

    def exist_row_by_datetime(self, coin_name, datetime):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(select_by_datetime.format("KRW_" + coin_name, datetime))

            row = cursor.fetchall()

            conn.commit()

            if len(row) == 0:
                return False
            else:

                return True

    def remove_duplicated_datetime(self, coin_name):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(remove_duplicated_datetime_sql.format("KRW_" + coin_name))
            conn.commit()


if __name__ == "__main__":
    upbit_recorder = UpbitRecorder()

    total_new_records = 0
    for coin_name in upbit_recorder.coin_names:
        total_new_records += upbit_recorder.record(coin_name)
        time.sleep(0.05)

    for coin_name in upbit_recorder.coin_names:
        upbit_recorder.remove_duplicated_datetime(coin_name)

    msg = "Number of new upbit records: {0} @ {1}".format(total_new_records, SOURCE)
    logger.info(msg)



