from pytz import timezone
import datetime as dt
import time
import traceback

import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.upbit_api import Upbit
from common.utils import convert_to_daily_timestamp
from common.logger import get_logger
from common.global_variables import *
from web.db.database import get_order_book_class

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

import warnings
warnings.filterwarnings('ignore')

logger = get_logger("upbit_order_book_recorder")

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("upbit"):
    os.chdir("..")

IS_INIT = False

engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')

db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

for coin_name in upbit.get_all_coin_names():
    if not engine.dialect.has_table(engine, "KRW_{0}_ORDER_BOOK".format(coin_name)):
        IS_INIT = True
        get_order_book_class(coin_name).__table__.create(bind=engine)
        print("KRW_{0}_ORDER_BOOK Table Created".format(coin_name))


class UpbitOrderBookRecorder:
    def __init__(self):
        self.coin_names = upbit.get_all_coin_names()

    def record(self, base_datetime, coin_ticker_name):
        daily_base_timestamp = convert_to_daily_timestamp(base_datetime)

        order_book = upbit.get_orderbook(tickers=coin_ticker_name)[0]

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
        for coin_name in order_book_info:
            order_book = get_order_book_class(coin_name)()
            order_book.base_datetime = order_book_info[coin_name]["base_datetime"]
            order_book.daily_base_timestamp = int(order_book_info[coin_name]["daily_base_timestamp"])
            order_book.collect_timestamp = int(order_book_info[coin_name]["collect_timestamp"])

            for idx in range(15):
                setattr(order_book, "ask_price_{0}".format(idx), order_book_info[coin_name]["ask_price_lst"][idx])
                setattr(order_book, "ask_size_{0}".format(idx), order_book_info[coin_name]["ask_size_lst"][idx])

                setattr(order_book, "bid_price_{0}".format(idx), order_book_info[coin_name]["bid_price_lst"][idx])
                setattr(order_book, "bid_size_{0}".format(idx), order_book_info[coin_name]["bid_size_lst"][idx])

            order_book.total_ask_size = order_book_info[coin_name]["total_ask_size"]
            order_book.total_bid_size = order_book_info[coin_name]["total_bid_size"]

            db_session.add(order_book)
            db_session.commit()

    def check_and_arrange_missing_order_book_data(self, coin_name, base_datetime_str):
        base_datetime = dt.datetime.strptime(base_datetime_str, fmt.replace("T", " "))

        missing_base_datetime_str_lst = []
        while base_datetime:
            base_datetime = base_datetime - dt.timedelta(minutes=10)
            base_datetime_str = dt.datetime.strftime(base_datetime, fmt.replace("T", " "))
            order_book_class = get_order_book_class(coin_name)
            exist = db_session.query(order_book_class).filter_by(base_datetime=base_datetime_str).scalar() is not None
            if exist:
                missing_base_datetime_str_lst.append(base_datetime_str)
                break
            else:
                missing_base_datetime_str_lst.append(base_datetime_str)

        if len(missing_base_datetime_str_lst) <= 1:
            return

        base_datetime_str = missing_base_datetime_str_lst[-1]
        q = db_session.query(order_book_class).filter_by(base_datetime=base_datetime_str)
        order_book = q.first()

        del missing_base_datetime_str_lst[-1]

        for missing_base_datetime_str in missing_base_datetime_str_lst:
            order_book.base_datetime = missing_base_datetime_str
            missing_daily_base_timestamp = convert_to_daily_timestamp(missing_base_datetime_str)
            order_book.daily_base_timestamp = missing_daily_base_timestamp
            order_book.collect_timestamp = -1
            db_session.add(order_book)
            db_session.commit()

            logger.info("{0}: missing_base_datetime: {1}".format(coin_name, missing_base_datetime_str))


if __name__ == "__main__":
    try:
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

        if not IS_INIT:
            for coin_name in upbit_order_book_recorder.coin_names:
                upbit_order_book_recorder.check_and_arrange_missing_order_book_data(coin_name, base_datetime)

    except Exception as ex:
        msg_str = "upbit_order_book_recorder.py - ERROR! \n"
        msg_str += ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        SLACK.send_message("me", msg_str)
