import datetime as dt
from datetime import timedelta

import sys, os

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from web.db.database import get_order_book_class

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.utils import convert_to_daily_timestamp
from common.logger import get_logger
from codes.upbit.upbit_api import Upbit
from common.global_variables import *

logger = get_logger("upbit_order_book_clean")

engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')

db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)


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

        final_base_datetime = dt.datetime.strptime(final_base_datetime_str, fmt.replace("T", " "))

        missing_count = 0
        while True:
            last_base_datetime_str = self.get_order_book_consecutiveness(
                start_base_datetime_str=start_base_datetime_str
            )

            if last_base_datetime_str == final_base_datetime_str:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2} --> FINAL".format(
                    self.coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                break

            if last_base_datetime_str is None:
                missing_count += 1
                logger.info("{0:5s} - Start Base Datetime: {1} - Missing: {2}".format(
                    self.coin_name, start_base_datetime_str, missing_count
                ))
                previous_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                previous_base_datetime = previous_base_datetime - dt.timedelta(minutes=10)
                previous_base_datetime_str = dt.datetime.strftime(previous_base_datetime, fmt.replace("T", " "))

                self.insert_missing_record(previous_base_datetime_str, start_base_datetime_str)

                start_base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=10)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))
            else:
                logger.info("{0:5s} - Start Base Datetime: {1}, Last Base Datetime: {2}".format(
                    self.coin_name, start_base_datetime_str, last_base_datetime_str
                ))
                start_base_datetime = dt.datetime.strptime(last_base_datetime_str, fmt.replace("T", " "))
                start_base_datetime = start_base_datetime + dt.timedelta(minutes=10)
                start_base_datetime_str = dt.datetime.strftime(start_base_datetime, fmt.replace("T", " "))

            if start_base_datetime >= final_base_datetime:
                break
        return missing_count, last_base_datetime_str

    def insert_missing_record(self, previous_base_datetime_str, missing_base_datetime_str):
        order_book_class = get_order_book_class(self.coin_name)
        exist_order_book = None
        try:
            exist_order_book = db_session.query(order_book_class).filter(order_book_class.base_datetime == previous_base_datetime_str).one()
        except sqlalchemy.orm.exc.MultipleResultsFound as e:
            print("Multiple rows were found for one()", e.__traceback__)
            print(self.coin_name, previous_base_datetime_str)
            exit()

        order_book_class = get_order_book_class(self.coin_name)
        exist = db_session.query(order_book_class).filter_by(base_datetime=missing_base_datetime_str).scalar() is not None
        if exist:
            return

        missing_order_book = order_book_class()
        missing_order_book.base_datetime = missing_base_datetime_str
        missing_order_book.daily_base_timestamp = convert_to_daily_timestamp(missing_base_datetime_str)
        missing_order_book.collect_timestamp = -1
        for idx in range(15):
            setattr(
                missing_order_book,
                "ask_price_{0}".format(idx),
                getattr(exist_order_book, "ask_price_{0}".format(idx))
            )
            setattr(
                missing_order_book,
                "ask_size_{0}".format(idx),
                getattr(exist_order_book, "ask_size_{0}".format(idx))
            )
            setattr(
                missing_order_book,
                "bid_price_{0}".format(idx),
                getattr(exist_order_book, "bid_price_{0}".format(idx))
            )
            setattr(
                missing_order_book,
                "bid_size_{0}".format(idx),
                getattr(exist_order_book, "bid_size_{0}".format(idx))
            )
        missing_order_book.total_ask_size = exist_order_book.total_ask_size
        missing_order_book.total_bid_size = exist_order_book.total_bid_size

        db_session.add(missing_order_book)
        db_session.commit()

    def get_order_book_start_and_final(self):
        order_book_class = get_order_book_class(self.coin_name)

        start_base_datetime_str = db_session.query(order_book_class.base_datetime).order_by(
            order_book_class.base_datetime.asc(), order_book_class.collect_timestamp.asc()).limit(1).one()[0]

        final_base_datetime_str = db_session.query(order_book_class.base_datetime).order_by(
            order_book_class.base_datetime.desc(), order_book_class.collect_timestamp.desc()).limit(1).one()[0]

        return start_base_datetime_str, final_base_datetime_str

    def get_order_book_consecutiveness(self, start_base_datetime_str=None):
        order_book_class = get_order_book_class(self.coin_name)

        if start_base_datetime_str is None:
            result = db_session.query(order_book_class.base_datetime).order_by(
                order_book_class.base_datetime.asc(), order_book_class.collect_timestamp.asc()
            ).limit(1).one_or_none()

            if result is None:
                return None
            else:
                start_base_datetime_str = result[0]

        result = db_session.query(order_book_class.base_datetime).filter(
            order_book_class.base_datetime == start_base_datetime_str
        ).one_or_none()

        if result is None:
            return None

        base_datetime = dt.datetime.strptime(start_base_datetime_str, fmt.replace("T", " "))
        last_base_datetime_str = None
        while True:
            next_base_datetime = base_datetime + timedelta(minutes=10)
            next_base_datetime_str = dt.datetime.strftime(next_base_datetime, fmt.replace("T", " "))

            result = db_session.query(order_book_class.base_datetime).filter(
                order_book_class.base_datetime == next_base_datetime_str
            ).one_or_none()

            if result is None:
                break
            else:
                next_base_datetime_str = result[0]

            last_base_datetime_str = next_base_datetime_str
            base_datetime = dt.datetime.strptime(next_base_datetime_str, fmt.replace("T", " "))

        return last_base_datetime_str


def make_arrangement(coin_names):
    for idx, coin_name in enumerate(coin_names):
        coin_order_book_arrangement = UpbitOrderBookArrangement(coin_name)
        missing_count, last_base_datetime_str = coin_order_book_arrangement.processing_missing_data()
        msg = "{0}, {1}: {2} Missing Data was Processed!. Last arranged data: {3}".format(
            idx,
            coin_name,
            missing_count,
            last_base_datetime_str
        )
        logger.info(msg)
        print("{0} {1} Completed".format(idx, coin_name))


if __name__ == "__main__":
    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    # 중요. BTC 데이터 부터 Missing_Data 처리해야 함.
    btc_order_book_arrangement = UpbitOrderBookArrangement("BTC")
    missing_count, last_base_datetime_str = btc_order_book_arrangement.processing_missing_data()
    msg = "{0}: {1} Missing Data was Processed!. Last arranged data: {2}".format(
        "BTC",
        missing_count,
        last_base_datetime_str
    )
    logger.info(msg)

    all_coin_names = upbit.get_all_coin_names()
    all_coin_names.remove("BTC")

    make_arrangement(all_coin_names)
