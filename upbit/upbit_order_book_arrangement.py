import datetime as dt
from datetime import timedelta

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.utils import convert_to_daily_timestamp
from common.logger import get_logger


from db.sqlite_handler import *

logger = get_logger("upbit_order_book_arrangement")


class UpbitOrderBookArrangement:
    def __init__(self, coin_name):
        self.coin_name = coin_name

    def processing_missing_data(self):
        logger.info("Processing Missing Data")

        start_base_datetime_str, final_base_datetime_str = self.get_order_book_start_and_final()
        logger.info("{0:5s} - Start: {1}, Final: {2}".format(
            self.coin_name,
            start_base_datetime_str,
            final_base_datetime_str
        ))

        missing_count = 0
        while True:
            last_base_datetime_str = self.get_order_book_consecutiveness(
                start_base_datetime_str=start_base_datetime_str
            )

            if last_base_datetime_str == final_base_datetime_str:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    self.coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                break

            if last_base_datetime_str is None:
                missing_count += 1
                logger.info("{0:5s} - Start Base Datetime: {1} - Missing: {2}".format(
                    self.coin_name, start_base_datetime_str, missing_count
                ))
                previous_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                previous_base_datetime = previous_base_datetime - dt.timedelta(minutes=1)
                previous_base_datetime_str = dt.datetime.strftime(previous_base_datetime, fmt.replace("T", " "))

                self.insert_missing_record(previous_base_datetime_str, start_base_datetime_str)

                start_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=1)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))
            else:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    self.coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                start_base_datetime = dt.datetime.strptime(last_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=1)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))

            if start_base_datetime_str == final_base_datetime_str:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    self.coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                break
        return missing_count, last_base_datetime_str

    def insert_missing_record(self, previous_base_datetime_str, missing_base_datetime_str):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(select_order_book_by_datetime.format(self.coin_name), (previous_base_datetime_str,))
            info = cursor.fetchone()

            cursor.execute(order_book_insert_sql.format(self.coin_name), (
                missing_base_datetime_str, convert_to_daily_timestamp(missing_base_datetime_str), info[3],
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

    def get_order_book_start_and_final(self):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()

            cursor.execute(select_by_start_base_datetime.format(self.coin_name))
            start_base_datetime_str = cursor.fetchone()[0]

            cursor.execute(select_by_final_base_datetime.format(self.coin_name))
            final_base_datetime_str = cursor.fetchone()[0]

            conn.commit()
        return start_base_datetime_str, final_base_datetime_str

    def get_order_book_consecutiveness(self, start_base_datetime_str=None):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()

            if start_base_datetime_str is None:
                cursor.execute(select_by_start_base_datetime.format(self.coin_name))
                start_base_datetime_str = cursor.fetchone()[0]

            cursor.execute(select_by_datetime.format(self.coin_name), (start_base_datetime_str,))
            start_base_datetime_str = cursor.fetchone()

            if start_base_datetime_str is None:
                return None

            base_datetime = dt.datetime.strptime(start_base_datetime_str[0], fmt.replace("T", " "))

            last_base_datetime_str = start_base_datetime_str[0]
            while True:
                next_base_datetime = base_datetime + timedelta(minutes=1)
                next_base_datetime_str = dt.datetime.strftime(next_base_datetime, fmt.replace("T", " "))

                cursor.execute(select_by_datetime.format(self.coin_name), (next_base_datetime_str, ))
                next_base_datetime_str = cursor.fetchone()

                if not next_base_datetime_str:
                    break

                next_base_datetime_str = next_base_datetime_str[0]
                last_base_datetime_str = next_base_datetime_str
                base_datetime = dt.datetime.strptime(next_base_datetime_str, fmt.replace("T", " "))

            conn.commit()

        return last_base_datetime_str


def make_arrangement(coin_names):
    # 중요. BTC 데이터 부터 Missing_Data 처리해야 함.
    btc_order_book_arrangement = UpbitOrderBookArrangement("BTC")
    missing_count, last_base_datetime_str = btc_order_book_arrangement.processing_missing_data()
    msg = "{0}: {1} Missing Data was Processed!. Last arranged data: {2}".format(
        "BTC",
        missing_count,
        last_base_datetime_str
    )
    logger.info(msg)

    for coin_name in coin_names:
        coin_order_book_arrangement = UpbitOrderBookArrangement(coin_name)
        missing_count, last_base_datetime_str = coin_order_book_arrangement.processing_missing_data()
        msg = "{0}: {1} Missing Data was Processed!. Last arranged data: {2}".format(
            coin_name,
            missing_count,
            last_base_datetime_str
        )
        logger.info(msg)


if __name__ == "__main__":
    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    make_arrangement(upbit.get_all_coin_names())

    # btc_order_book_arrangement = UpbitOrderBookArrangement("DCR")
    # missing_count, last_base_datetime_str = btc_order_book_arrangement.processing_missing_data()
    # msg = "{0}: {1} Missing Data was Processed!. Last arranged data: {2}".format(
    #     "BTC",
    #     missing_count,
    #     last_base_datetime_str
    # )
    # logger.info(msg)
