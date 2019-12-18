import sys, os
import time

from codes.upbit.recorder.upbit_price_collector import Unit, get_next_date_time

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.upbit_api import Upbit
from common.logger import get_logger
from common.global_variables import *
from web.db.database import get_order_book_class

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import warnings
warnings.filterwarnings('ignore')

logger = get_logger("upbit_order_book_transfer")

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("upbit"):
    os.chdir("..")

IS_INIT = False

mysql_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/record'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')

mysql_db_session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=mysql_engine)
        )


sqlite_engine = create_engine('sqlite:///{0}/web/db/upbit_order_book_info.db'.format(
            PROJECT_HOME
        ))

sqlite_db_session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=sqlite_engine)
        )

coin_names = upbit.get_all_coin_names()

# with mysql_engine.connect() as con:
#     for c_idx, coin_name in enumerate(coin_names):
#         con.execute('ALTER TABLE KRW_{0}_ORDER_BOOK MODIFY `collect_timestamp` bigint;'.format(coin_name))
#         print(c_idx, coin_name)




for coin_name in coin_names:
    order_book_class = get_order_book_class(coin_name)
    sqlite_order_books = sqlite_db_session.query(order_book_class).order_by(order_book_class.base_datetime.asc()).all()
    print(coin_name)
    logger.info(coin_name)

    expected_base_datetime = None
    is_first = True

    for idx, sqlite_order_book in enumerate(sqlite_order_books):
        ### Check missing record
        if not is_first and expected_base_datetime != str(sqlite_order_book.base_datetime):
            print(expected_base_datetime, str(sqlite_order_book.base_datetime))
            logger.info("{0} is missing at sqlite_order_book".format(expected_base_datetime))

        expected_base_datetime = get_next_date_time(
            sqlite_order_book.base_datetime,
            unit=Unit.TEN_MINUTES,
            count=1
        )

        is_first = False

        #### Transfer data
        q = mysql_db_session.query(order_book_class).filter(order_book_class.base_datetime == sqlite_order_book.base_datetime)
        stored_order_book = q.first()

        if stored_order_book is None:
            local_object = mysql_db_session.merge(sqlite_order_book)
            mysql_db_session.add(local_object)
            logger.info("{0} data is stored to mysql database.".format(local_object.base_datetime))
            mysql_db_session.commit()
        else:
            logger.info("{0} data already exist at mysql database.".format(stored_order_book.base_datetime))

        if idx % 100 == 0:
            print(idx, end=", ")
            sys.stdout.flush()

    print(idx)
    sys.stdout.flush()
    mysql_db_session.commit()
    sqlite_db_session.commit()



