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

######################
trade_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8', pool_recycle=500, pool_size=5, max_overflow=20, echo=False, echo_pool=True)

trade_db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=trade_engine)
)

######################
buy_sell_engine = create_engine('sqlite:///{0}/web/db/upbit_buy_sell.db'.format(PROJECT_HOME))
buy_sell_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=buy_sell_engine))


######################
naver_order_book_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/record_order_book?use_pure=True'.format(
            NAVER_MYSQL_ID, NAVER_MYSQL_PASSWORD, NAVER_MYSQL_HOST
        ), encoding='utf-8', pool_recycle=500, pool_size=5, max_overflow=20, echo=False, echo_pool=True)
naver_order_book_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=naver_order_book_engine))


class Model(Base):
    __tablename__ = 'MODEL'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(16))
    model_filename = Column(String(32))
    window_size = Column(Integer)
    future_target_size = Column(Integer)
    up_rate = Column(Float)
    feature_size = Column(Integer)
    datetime = Column(String(32))
    one_rate = Column(Float)
    train_size = Column(Integer)
    best_score = Column(Float)

    def to_json(self):
        return {
            "model_type": self.model_type,
            "window_size": self.window_size,
            "future_target_size": self.future_target_size,
            "up_rate": self.up_rate,
            "feature_size": self.feature_size,
            "one_rate": self.one_rate,
            "train_size": self.train_size,
            "best_score": self.best_score,
            "datetime": self.datetime
        }

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


class UpbitInfo(Base):
    __tablename__ = "UPBIT_INFO"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(16))
    korean_name = Column(String(16))
    eng_name = Column(String(16))
    limit_amount_max = Column(Float)
    limit_amount_min = Column(Float)
    limit_cost_max = Column(Float)
    limit_cost_min = Column(Float)
    limit_price_max = Column(Float)
    limit_price_min = Column(Float)
    maker = Column(Float)
    taker = Column(Float)
    percentage = Column(Boolean)
    precision_amount = Column(Float)
    precision_price = Column(Float)
    tierBased = Column(Boolean)

    birth = Column(String(64))
    total_markets = Column(String(64))
    num_exchanges = Column(String(64))
    period_block_creation = Column(String(64))
    mine_reward_unit = Column(String(64))
    total_limit = Column(String(64))
    consensus_protocol = Column(String(64))
    web_site = Column(String(256))
    whitepaper = Column(String(256))
    block_site = Column(String(256))
    twitter_url = Column(String(256))
    intro = Column(String(1024))

    def __init__(self, *args, **kw):
        super(UpbitInfo, self).__init__(*args, **kw)

    def get_id(self):
        return self.id

    def to_dict(self):
        d = {}
        d["coin_name"] = self.market.replace("KRW-", "")
        d["market"] = self.market
        d["korean_name"] = self.korean_name
        d["eng_name"] = self.eng_name
        d["limit_amount_max"] = self.limit_amount_max
        d["limit_amount_min"] = self.limit_amount_min
        d["limit_cost_max"] = self.limit_cost_max
        d["limit_cost_min"] = self.limit_cost_min
        d["limit_price_max"] = self.limit_price_max
        d["limit_price_min"] = self.limit_price_min
        d["maker"] = self.maker
        d["taker"] = self.taker
        d["percentage"] = self.percentage
        d["precision_amount"] = self.precision_amount
        d["precision_price"] = self.precision_price
        d["tierBased"] = self.tierBased

        d["birth"] = self.birth
        d["total_markets"] = self.total_markets
        d["num_exchanges"] = self.num_exchanges
        d["period_block_creation"] = self.period_block_creation
        d["mine_reward_unit"] = self.mine_reward_unit
        d["total_limit"] = self.total_limit
        d["consensus_protocol"] = self.consensus_protocol
        d["web_site"] = self.web_site
        d["whitepaper"] = self.whitepaper
        d["block_site"] = self.block_site
        d["twitter_url"] = self.twitter_url

        d["intro"] = self.intro

        return d


if __name__ == "__main__":
    if not buy_sell_engine.dialect.has_table(buy_sell_engine, "BUY_SELL"):
        BuySell.__table__.create(bind=buy_sell_engine)
        print("BUY_SELL Table Created")

    if not trade_engine.dialect.has_table(trade_engine, "MODEL"):
        Model.__table__.create(bind=trade_engine)
        print("MODEL Table Created")

    if not trade_engine.dialect.has_table(trade_engine, "UPBIT_INFO"):
        UpbitInfo.__table__.create(bind=trade_engine)
        print("UPBIT_NFO Table Created")

    if not trade_engine.dialect.has_table(trade_engine, "USER"):
        User.__table__.create(bind=trade_engine)
        print("USER Table Created")

