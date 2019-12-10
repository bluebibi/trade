import datetime
import pprint
import time

import pytz
import requests, json
import os, sys
import enum

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt, MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

local_fmt = "%Y-%m-%d %H:%M:%S"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

utc_date_time_example = "2017-09-27 00:10:00"


class Unit(enum.Enum):
    # ONE_MINUTE = "1_MIN"
    TEN_MINUTES = "10_MIN"
    ONE_HOUR = "1_HOUR"
    ONE_DAY = "1_DAY"


url_list = {
    # Unit.ONE_MINUTE: 'https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/1?code=CRIX.UPBIT.KRW-{0}&count={1}&to={2}',
    Unit.TEN_MINUTES: 'https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/10?code=CRIX.UPBIT.KRW-{0}&count={1}&to={2}',
    Unit.ONE_HOUR: 'https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/60?code=CRIX.UPBIT.KRW-{0}&count={1}&to={2}',
    Unit.ONE_DAY: 'https://crix-api-endpoint.upbit.com/v1/crix/candles/days?code=CRIX.UPBIT.KRW-{0}&count={1}&to={2}'
}


def convert_utc_to_seoul_time(date_time_utc):
    local_timezone = pytz.timezone('Asia/Seoul')
    date_time_utc = datetime.datetime.strptime(date_time_utc, local_fmt)
    local_time = date_time_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
    return local_time.strftime(local_fmt)


def get_next_date_time(date_time, unit=Unit.TEN_MINUTES, count=1):
    '''

    :param start: "2017-09-27 00:10:00"
    :param unit: "1min"
    :param count: 2
    :return: "2017-09-27 00:12:00"
    '''
    date_time = datetime.datetime.strptime(date_time, local_fmt)

    if unit == Unit.TEN_MINUTES:
        next_date_time = date_time + datetime.timedelta(minutes=10 * count)
    elif unit == Unit.ONE_HOUR:
        next_date_time = date_time + datetime.timedelta(hours=1 * count)
    elif unit == Unit.ONE_DAY:
        next_date_time = date_time + datetime.timedelta(days=1 * count)
    else:
        raise ValueError("{0} unit is not supported".format(unit))

    return next_date_time.strftime(local_fmt)

def get_next_one_second_date_time(date_time):
    date_time = datetime.datetime.strptime(date_time, local_fmt)
    previous_date_time = date_time + datetime.timedelta(seconds=1)
    return previous_date_time.strftime(local_fmt)


def get_previous_one_second_date_time(date_time):
    date_time = datetime.datetime.strptime(date_time, local_fmt)
    previous_date_time = date_time - datetime.timedelta(seconds=1)
    return previous_date_time.strftime(local_fmt)

def get_previous_date_time(date_time, unit=Unit.TEN_MINUTES, count=1):
    '''

    :param start: "2017-09-27 00:10:00"
    :param unit: Unit.TEN_MINUTES
    :param count: 2
    :return: "2017-09-26 23:50:00"
    '''
    date_time = datetime.datetime.strptime(date_time, local_fmt)

    if unit == Unit.TEN_MINUTES:
        previous_date_time = date_time - datetime.timedelta(minutes=10 * count)
    elif unit == Unit.ONE_HOUR:
        previous_date_time = date_time - datetime.timedelta(hours=1 * count)
    elif unit == Unit.ONE_DAY:
        previous_date_time = date_time - datetime.timedelta(days=1 * count)
    else:
        raise ValueError("{0} unit is not supported".format(unit))

    return previous_date_time.strftime(local_fmt)


def get_utc_now_string():
    return datetime.datetime.utcnow().strftime(local_fmt)


def get_last_date_time(date_time, unit):
    if unit == Unit.ONE_DAY:
        if date_time.endswith("00:00:00"):
            return date_time
        date_time = datetime.datetime.strptime(date_time, local_fmt)
        while True:
            date_time = date_time - datetime.timedelta(seconds=1)
            date_time_str = date_time.strftime(local_fmt)
            if date_time_str.endswith("00:00:00"):
                break
    elif unit == Unit.ONE_HOUR:
        if date_time.endswith("00:00"):
            return date_time
        date_time = datetime.datetime.strptime(date_time, local_fmt)
        while True:
            date_time = date_time - datetime.timedelta(seconds=1)
            date_time_str = date_time.strftime(local_fmt)
            if date_time_str.endswith("00:00"):
                break
    elif unit == Unit.TEN_MINUTES:
        if date_time.endswith("0:00"):
            return date_time
        date_time = datetime.datetime.strptime(date_time, local_fmt)
        while True:
            date_time = date_time - datetime.timedelta(seconds=1)
            date_time_str = date_time.strftime(local_fmt)
            if date_time_str.endswith("0:00"):
                break
    else:
        raise ValueError("{0} unit is not supported".format(unit))

    return date_time_str


def get_price(coin_name, to_datetime, unit, count):
    fail_get_data = False
    fail_count = 0
    text = None
    while True:
        try:
            text = requests.get(
                url_list[unit].format(coin_name, count, to_datetime),
                headers=headers
            ).text   # url에 있는 데이터 읽기
        except Exception as e:
            print('Error code: ', e, ' 5초 대기')
            fail_count += 1
            if fail_count > 10:
                fail_get_data = True
                break
            time.sleep(1)
        else:
            break
    if fail_get_data:
        print("Fail to access url")
        exit()
    data = json.loads(text)    # json 구조로 변환

    price_list = []
    last_utc_date_time = None
    for i in range(len(data)):
        utc_date = data[i]['candleDateTime']
        date = data[i]['candleDateTimeKst']
        price_list.append(
            [
                i,
                utc_date,
                date,
                "%f" % data[i]['openingPrice'],
                "%f" % data[i]['highPrice'],
                "%f" % data[i]['lowPrice'],
                "%f" % data[i]['tradePrice'],
                "%f" % data[i]['candleAccTradeVolume']
            ]
        )
        last_utc_date_time = datetime.datetime.strptime(utc_date.split("+")[0], "%Y-%m-%dT%H:%M:%S")

    return price_list, last_utc_date_time.strftime(local_fmt)


mysql_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/trade'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')
Base = declarative_base()


def get_coin_price_class(coin_name, unit):
    class CoinPrice(Base):
        __tablename__ = "KRW_{0}_PRICE_{1}".format(coin_name, unit.value)
        __table_args__ = {'extend_existing': True}

        id = Column(Integer, primary_key=True, autoincrement=True)
        datetime_utc = Column(DateTime)
        datetime_krw = Column(DateTime)
        open = Column(Float)
        high = Column(Float)
        low = Column(Float)
        final = Column(Float)
        volume = Column(Float)

    return CoinPrice


Base.metadata.create_all(mysql_engine)
db_session = sessionmaker(bind=mysql_engine)
db_session = db_session()

for coin_name in upbit.get_all_coin_names():
    for unit in Unit:
        if not mysql_engine.dialect.has_table(mysql_engine, "KRW_{0}_PRICE_{1}".format(coin_name, unit.value)):
            get_coin_price_class(coin_name, unit).__table__.create(bind=mysql_engine)


def collect(unit, idx, coin_name, count):
    coin_price_class = get_coin_price_class(coin_name, unit)

    to_utc_date_time = get_utc_now_string()
    to_utc_date_time = get_last_date_time(to_utc_date_time, unit)
    to_utc_date_time = get_next_one_second_date_time(to_utc_date_time)

    price_list, _ = get_price(coin_name, to_utc_date_time, unit, count)

    utc_date_time_first_inserted = None
    utc_date_time_last_inserted = None

    is_first = True
    for price in price_list:
        datetime_utc = price[1].split("+")[0].replace("T", " ")
        datetime_krw = price[2].split("+")[0].replace("T", " ")
        q = db_session.query(coin_price_class).filter(coin_price_class.datetime_utc == datetime_utc)
        coin_price = q.first()
        if coin_price is None:
            coin_price = coin_price_class()
            coin_price.datetime_utc = datetime_utc
            coin_price.datetime_krw = datetime_krw
            coin_price.open = float(price[3])
            coin_price.high = float(price[4])
            coin_price.low = float(price[5])
            coin_price.final = float(price[6])
            coin_price.volume = float(price[7])

            db_session.add(coin_price)
            db_session.commit()
            utc_date_time_last_inserted = datetime_utc
            if is_first:
                utc_date_time_first_inserted = datetime_utc
                is_first = False

    print("[{0}-{1}-{2}] First Inserted: {3}, Last Inserted: {4}".format(
        unit, idx, coin_name, utc_date_time_first_inserted, utc_date_time_last_inserted
    ))
    return utc_date_time_first_inserted, utc_date_time_last_inserted


def fill_missing_data(unit, coin_name, utc_date_time_first_inserted, utc_date_time_last_inserted):
    coin_price_class = get_coin_price_class(coin_name, unit)

    utc_date_time_first_inserted = datetime.datetime.strptime(utc_date_time_first_inserted, local_fmt)
    utc_date_time_last_inserted = datetime.datetime.strptime(utc_date_time_last_inserted, local_fmt)
    coin_price_list = db_session.query(coin_price_class).filter(
        and_(
            utc_date_time_first_inserted >= coin_price_class.datetime_utc,
            coin_price_class.datetime_utc >= utc_date_time_last_inserted
        )
    ).order_by(coin_price_class.datetime_utc.asc()).all()

    for coin_price in coin_price_list:
        if str(coin_price.datetime_utc) == str(utc_date_time_first_inserted):
            print(str(coin_price.datetime_utc), ":", str(utc_date_time_first_inserted))
            break

        next_datetime = get_next_date_time(str(coin_price.datetime_utc), unit, 1)

        next_coin_price = db_session.query(coin_price_class).filter(coin_price_class.datetime_utc == next_datetime).first()

        if next_coin_price is None:
            new_coin_price = coin_price_class()
            new_coin_price.datetime_utc = next_datetime
            new_coin_price.datetime_krw = convert_utc_to_seoul_time(next_datetime)
            new_coin_price.open = coin_price.open
            new_coin_price.high = coin_price.high
            new_coin_price.low = coin_price.low
            new_coin_price.final = coin_price.final
            new_coin_price.volume = coin_price.volume
            db_session.add(new_coin_price)
            db_session.commit()
            print("[{0}-{1}-{2}] Missing Price Info Inserted: {3}".format(unit, idx, coin_name, next_datetime))


if __name__ == "__main__":
    count = 100
    for unit in Unit:
        for idx, coin_name in enumerate(upbit.get_all_coin_names()):
            utc_date_time_first_inserted, utc_date_time_last_inserted = collect(unit, idx, coin_name, count)
            if utc_date_time_first_inserted and utc_date_time_last_inserted:
                fill_missing_data(unit, coin_name, utc_date_time_first_inserted, utc_date_time_last_inserted)