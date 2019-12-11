from datetime import datetime
from enum import Enum

import pytz
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class CoinStatus(Enum):
    bought = 0
    trailed = 1
    success_sold = 2
    gain_sold = 3
    loss_sold = 4
    up_trailed = 5


def get_order_book_class(coin_name):
   class OrderBook(db.Model):
        __bind_key__ = 'upbit_order_book_info'
        __tablename__ = "KRW_{0}_ORDER_BOOK".format(coin_name)
        __table_args__ = {'extend_existing': True}

        id = db.Column(db.Integer, primary_key=True, autoincrement=True)
        base_datetime = db.Column(db.DateTime, unique=True, index=True)
        daily_base_timestamp = db.Column(db.Integer)
        from sqlalchemy.dialects.mysql import BIGINT
        collect_timestamp = db.Column(BIGINT(unsigned=True), index=True)

        ask_price_0 = db.Column(db.Float)
        ask_size_0 = db.Column(db.Float)

        ask_price_1 = db.Column(db.Float)
        ask_size_1 = db.Column(db.Float)

        ask_price_2 = db.Column(db.Float)
        ask_size_2 = db.Column(db.Float)

        ask_price_3 = db.Column(db.Float)
        ask_size_3 = db.Column(db.Float)

        ask_price_4 = db.Column(db.Float)
        ask_size_4 = db.Column(db.Float)

        ask_price_5 = db.Column(db.Float)
        ask_size_5 = db.Column(db.Float)

        ask_price_6 = db.Column(db.Float)
        ask_size_6 = db.Column(db.Float)
        
        ask_price_7 = db.Column(db.Float)
        ask_size_7 = db.Column(db.Float)
        
        ask_price_8 = db.Column(db.Float)
        ask_size_8 = db.Column(db.Float)
        
        ask_price_9 = db.Column(db.Float)
        ask_size_9 = db.Column(db.Float)
        
        ask_price_10 = db.Column(db.Float)
        ask_size_10 = db.Column(db.Float)
        
        ask_price_11 = db.Column(db.Float)
        ask_size_11 = db.Column(db.Float)
        
        ask_price_12 = db.Column(db.Float)
        ask_size_12 = db.Column(db.Float)
        
        ask_price_13 = db.Column(db.Float)
        ask_size_13 = db.Column(db.Float)
        
        ask_price_14 = db.Column(db.Float)
        ask_size_14 = db.Column(db.Float)
        
        bid_price_0 = db.Column(db.Float)
        bid_size_0 = db.Column(db.Float)
        
        bid_price_1 = db.Column(db.Float)
        bid_size_1 = db.Column(db.Float)
        
        bid_price_2 = db.Column(db.Float)
        bid_size_2 = db.Column(db.Float)
        
        bid_price_3 = db.Column(db.Float)
        bid_size_3 = db.Column(db.Float)
        
        bid_price_4 = db.Column(db.Float)
        bid_size_4 = db.Column(db.Float)
        
        bid_price_5 = db.Column(db.Float)
        bid_size_5 = db.Column(db.Float)
        
        bid_price_6 = db.Column(db.Float)
        bid_size_6 = db.Column(db.Float)
        
        bid_price_7 = db.Column(db.Float)
        bid_size_7 = db.Column(db.Float)
        
        bid_price_8 = db.Column(db.Float)
        bid_size_8 = db.Column(db.Float)
        
        bid_price_9 = db.Column(db.Float)
        bid_size_9 = db.Column(db.Float)
        
        bid_price_10 = db.Column(db.Float)
        bid_size_10 = db.Column(db.Float)
        
        bid_price_11 = db.Column(db.Float)
        bid_size_11 = db.Column(db.Float)
        
        bid_price_12 = db.Column(db.Float)
        bid_size_12 = db.Column(db.Float)
        
        bid_price_13 = db.Column(db.Float)
        bid_size_13 = db.Column(db.Float)
        
        bid_price_14 = db.Column(db.Float)
        bid_size_14 = db.Column(db.Float)
        
        total_ask_size = db.Column(db.Float)
        total_bid_size = db.Column(db.Float)

        def __init__(self, *args, **kw):
            super(OrderBook, self).__init__(*args, **kw)

        def get_id(self):
            return self.id

        def __repr__(self):
            return str({
                "id": self.id,
                "base_datetime": self.base_datetime,
                "ask_price_0": self.ask_size_0,
                "bid_price_0": self.bid_price_0
            })

   return OrderBook


class BuySell(db.Model):
    __bind_key__ = 'upbit_buy_sell'
    __tablename__ = "BUY_SELL"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    coin_ticker_name = db.Column(db.String(8), index=True)
    buy_datetime = db.Column(db.String(32), index=True)
    lstm_prob = db.Column(db.Float)
    gb_prob = db.Column(db.Float)
    xgboost_prob = db.Column(db.Float)
    buy_base_price = db.Column(db.Float)
    buy_krw = db.Column(db.Integer)
    buy_fee = db.Column(db.Integer)
    buy_price = db.Column(db.Float)
    buy_coin_volume = db.Column(db.Float)
    trail_datetime = db.Column(db.String(32))
    trail_price = db.Column(db.Float)
    sell_fee = db.Column(db.Integer)
    sell_krw = db.Column(db.Integer)
    trail_rate = db.Column(db.Float)
    total_krw = db.Column(db.Integer)
    trail_up_count = db.Column(db.Integer)
    status = db.Column(db.Integer)
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


class User(db.Model):
    __bind_key__ = 'user'
    __tablename__ = 'USER'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.Unicode(128), nullable=False)
    name = db.Column(db.Unicode(128))
    password = db.Column(db.Unicode(128))

    created_on_datetime = datetime.now(pytz.timezone('Asia/Seoul'))
    created_on = db.Column(db.DateTime, default=created_on_datetime)
    created_on_str = db.Column(db.Unicode(128), default=created_on_datetime.strftime('%Y년 %m월 %d일 %H시 %M분'))

    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)

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