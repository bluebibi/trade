import os, sys

from sqlalchemy import func

from codes.upbit.recorder.price_collect import get_coin_price_class, db_session, Unit, local_fmt, get_price

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if __name__ == "__main__":
    count = 200
    for unit in Unit:
        num_of_total_collect = 0
        while True:
            for idx, coin_name in enumerate(upbit.get_all_coin_names()):
                coin_price_class = get_coin_price_class(coin_name, unit)
                last_utc_date_time = db_session.query(func.min(coin_price_class.datetime_utc)).one()[0]

                last_utc_date_time = last_utc_date_time.strftime(local_fmt)

                price_list, _ = get_price(coin_name, last_utc_date_time, unit, count)
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

                num_of_total_collect += len(price_list)
                print(unit, idx, coin_name, last_utc_date_time, len(price_list))

            if num_of_total_collect == 0:
                break
