from datetime import datetime
from enum import Enum
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import pytz
from sqlalchemy import DateTime, Column, Integer, Float, String, Unicode, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.mysql import BIGINT

from common.global_variables import MYSQL_HOST, MYSQL_PASSWORD, MYSQL_ID, NAVER_MYSQL_ID, NAVER_MYSQL_PASSWORD, \
    NAVER_MYSQL_HOST

Base = declarative_base()

class CoinStatus(Enum):
    bought = 0
    trailed = 1
    success_sold = 2
    gain_sold = 3
    loss_sold = 4
    up_trailed = 5


buy_sell_engine = create_engine('sqlite:///{0}/web/db/upbit_buy_sell.db'.format(PROJECT_HOME))
buy_sell_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=buy_sell_engine))

naver_order_book_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/record_order_book?use_pure=True'.format(
            NAVER_MYSQL_ID, NAVER_MYSQL_PASSWORD, NAVER_MYSQL_HOST
        ), encoding='utf-8')
naver_order_book_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=naver_order_book_engine))

model_engine = create_engine('sqlite:///{0}/web/db/model.db'.format(PROJECT_HOME))
model_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=model_engine))

user_engine = create_engine('sqlite:///{0}/web/db/user.db'.format(PROJECT_HOME))
user_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=user_engine))

upbit_info_engine = create_engine(
    'sqlite:///{0}/web/db/upbit_info.db'.format(PROJECT_HOME),
    echo=False, connect_args={'check_same_thread': False}
)
upbit_info_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=upbit_info_engine))

class Model(Base):
    __tablename__ = 'MODEL'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    model_type = Column(String)
    model_filename = Column(String)
    window_size = Column(Integer)
    future_target_size = Column(Integer)
    up_rate = Column(Float)
    feature_size = Column(Integer)
    datetime = Column(String)
    one_rate = Column(Float)
    train_size = Column(Integer)
    best_score = Column(Float)

    def __repr__(self):
        return "[ID: {0}] Model Type: {1}, Model Filename: {2}, Datetime: {3}, One Rate: {4}, Train Size: {5}".format(
            self.id, self.model_type, self.model_filename, self.datetime, self.one_rate, self.train_size
        )


def get_order_book_class(coin_name):
   class OrderBook(Base):
        __tablename__ = "KRW_{0}_ORDER_BOOK".format(coin_name)
        __table_args__ = {'extend_existing': True}

        id = Column(Integer, primary_key=True, autoincrement=True)
        base_datetime = Column(DateTime, unique=True, index=True)
        daily_base_timestamp = Column(Integer)
        collect_timestamp = Column(BIGINT, index=True)

        ask_price_0 = Column(Float)
        ask_size_0 = Column(Float)

        ask_price_1 = Column(Float)
        ask_size_1 = Column(Float)

        ask_price_2 = Column(Float)
        ask_size_2 = Column(Float)

        ask_price_3 = Column(Float)
        ask_size_3 = Column(Float)

        ask_price_4 = Column(Float)
        ask_size_4 = Column(Float)

        ask_price_5 = Column(Float)
        ask_size_5 = Column(Float)

        ask_price_6 = Column(Float)
        ask_size_6 = Column(Float)

        ask_price_7 = Column(Float)
        ask_size_7 = Column(Float)

        ask_price_8 = Column(Float)
        ask_size_8 = Column(Float)

        ask_price_9 = Column(Float)
        ask_size_9 = Column(Float)

        ask_price_10 = Column(Float)
        ask_size_10 = Column(Float)

        ask_price_11 = Column(Float)
        ask_size_11 = Column(Float)

        ask_price_12 = Column(Float)
        ask_size_12 = Column(Float)

        ask_price_13 = Column(Float)
        ask_size_13 = Column(Float)

        ask_price_14 = Column(Float)
        ask_size_14 = Column(Float)

        bid_price_0 = Column(Float)
        bid_size_0 = Column(Float)

        bid_price_1 = Column(Float)
        bid_size_1 = Column(Float)

        bid_price_2 = Column(Float)
        bid_size_2 = Column(Float)

        bid_price_3 = Column(Float)
        bid_size_3 = Column(Float)

        bid_price_4 = Column(Float)
        bid_size_4 = Column(Float)

        bid_price_5 = Column(Float)
        bid_size_5 = Column(Float)

        bid_price_6 = Column(Float)
        bid_size_6 = Column(Float)

        bid_price_7 = Column(Float)
        bid_size_7 = Column(Float)

        bid_price_8 = Column(Float)
        bid_size_8 = Column(Float)

        bid_price_9 = Column(Float)
        bid_size_9 = Column(Float)

        bid_price_10 = Column(Float)
        bid_size_10 = Column(Float)

        bid_price_11 = Column(Float)
        bid_size_11 = Column(Float)

        bid_price_12 = Column(Float)
        bid_size_12 = Column(Float)

        bid_price_13 = Column(Float)
        bid_size_13 = Column(Float)

        bid_price_14 = Column(Float)
        bid_size_14 = Column(Float)

        total_ask_size = Column(Float)
        total_bid_size = Column(Float)

        def __init__(self, *args, **kw):
            super(OrderBook, self).__init__(*args, **kw)

        def get_id(self):
            return self.id

        def __repr__(self):
            return str({
                "id": self.id,
                "base_datetime": self.base_datetime,
                "ask_price_0": self.ask_price_0,
                "bid_price_0": self.bid_price_0
            })

   return OrderBook


class BuySell(Base):
    __tablename__ = "BUY_SELL"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_ticker_name = Column(String(8), index=True)
    buy_datetime = Column(String(32), index=True)
    lstm_prob = Column(Float)
    gb_prob = Column(Float)
    xgboost_prob = Column(Float)
    buy_base_price = Column(Float)
    buy_krw = Column(Integer)
    buy_fee = Column(Integer)
    buy_price = Column(Float)
    buy_coin_volume = Column(Float)
    trail_datetime = Column(String(32))
    trail_price = Column(Float)
    sell_fee = Column(Integer)
    sell_krw = Column(Integer)
    trail_rate = Column(Float)
    total_krw = Column(Integer)
    trail_up_count = Column(Integer)
    status = Column(Integer)
    elapsed_time = None
    coin_status = None

    def __init__(self, *args, **kw):
        super(BuySell, self).__init__(*args, **kw)

    def get_id(self):
        return self.id

    def to_json(self):
        return {
            "buy_datetime": self.buy_datetime,
            "coin_ticker_name": self.coin_ticker_name,
            "gb_prob": self.gb_prob,
            "xgboost_prob": self.xgboost_prob,
            "buy_base_price": self.buy_base_price,
            "buy_price": self.buy_price,
            "trail_price": self.trail_price,
            "buy_krw": self.buy_krw,
            "sell_krw": self.sell_krw,
            "elapsed_time": self.elapsed_time,
            "trail_rate": self.trail_rate,
            "coin_status": self.coin_status
        }


class User(Base):
    __tablename__ = 'USER'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(Unicode(128), nullable=False)
    name = Column(Unicode(128))
    password = Column(Unicode(128))

    created_on_datetime = datetime.now(pytz.timezone('Asia/Seoul'))
    created_on = Column(DateTime, default=created_on_datetime)
    created_on_str = Column(Unicode(128), default=created_on_datetime.strftime('%Y년 %m월 %d일 %H시 %M분'))

    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    def __init__(self, *args, **kw):
        super(User, self).__init__(*args, **kw)
        self._authenticated = False

    def set_password(self, password):
        self.password = generate_password_hash(password)

    @property
    def is_authenticated(self):
        return self._authenticated

    def authenticate(self, password):
        checked = check_password_hash(self.password, password)
        self._authenticated = checked
        return self._authenticated

    def get_id(self):
        return self.id

    def to_json(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_on': self.created_on_str,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
        }

    def __repr__(self):
        r = {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_on': self.created_on_str,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
        }
        return str(r)


if __name__ == "__main__":
    if not buy_sell_engine.dialect.has_table(buy_sell_engine, "BUY_SELL"):
        BuySell.__table__.create(bind=buy_sell_engine)
        print("BUY_SELL Table Created")

    if not model_engine.dialect.has_table(model_engine, "MODEL"):
        Model.__table__.create(bind=model_engine)
        print("MODEL Table Created")

    if not user_engine.dialect.has_table(user_engine, "USER"):
        User.__table__.create(bind=user_engine)
        print("USER Table Created")