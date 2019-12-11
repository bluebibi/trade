import datetime as dt
from datetime import timedelta

import sys, os

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
import sqlite3

logger = get_logger("upbit_order_book_clean")

engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')

db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)


if __name__ == "__main__":
    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    for coin_name in upbit.get_all_coin_names():
        order_book_class = get_order_book_class(coin_name)

