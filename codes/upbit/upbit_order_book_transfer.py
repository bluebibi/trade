import sys, os

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

logger = get_logger("upbit_order_book_recorder")

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("upbit"):
    os.chdir("..")

IS_INIT = False

mysql_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
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

for coin_name in coin_names:
    order_book_class = get_order_book_class(coin_name)
    order_books = sqlite_db_session.query(order_book_class).all()
    print(coin_name)
    for idx, order_book in enumerate(order_books):
        q = mysql_db_session.query(order_book_class).filter(order_book_class.collect_timestamp == order_book.collect_timestamp)
        stored_order_book = q.first()
        if stored_order_book is None:
            local_object = mysql_db_session.merge(order_book)
            mysql_db_session.add(local_object)
            print(idx, end=", ")
    mysql_db_session.commit()
    sqlite_db_session.commit()



