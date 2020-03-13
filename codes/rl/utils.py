from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt
from common.utils import convert_unit_4, convert_unit_8, convert_unit_1
import datetime as dt
import pprint
pp = pprint.PrettyPrinter(indent=2)


def get_buying_price_by_order_book(readyset_krw_price, order_book):
    commission_fee = int(readyset_krw_price * 0.0015)
    buying_krw_price = readyset_krw_price - commission_fee
    bought_coin_quantity = 0.0
    for order in order_book['orderbook_units']:
        units_ask = convert_unit_4(buying_krw_price / float(order['ask_price']))
        if units_ask <= float(order['ask_size']):
            bought_coin_quantity += units_ask
            buying_krw_price = buying_krw_price - float(order['ask_price']) * units_ask
            break
        else:
            buying_krw_price = buying_krw_price - float(order['ask_price']) * float(order['ask_size'])
            bought_coin_quantity += float(order['ask_size'])

    bought_coin_quantity = convert_unit_8(bought_coin_quantity)
    bought_krw_price = round(readyset_krw_price - buying_krw_price - commission_fee)
    bought_coin_price = convert_unit_1(bought_krw_price / bought_coin_quantity)

    return bought_krw_price, bought_coin_price, bought_coin_quantity, commission_fee


def get_selling_price_by_order_book(readyset_quantity, order_book):
    selling_quantity = readyset_quantity
    selling_krw_price = 0.0
    for order in order_book['orderbook_units']:
        if selling_quantity <= float(order['bid_size']):
            selling_krw_price += order['bid_price'] * selling_quantity
            break
        else:
            selling_quantity = selling_quantity - float(order['bid_size'])
            selling_krw_price += order['bid_price'] * float(order['bid_size'])

    commission_fee = round(selling_krw_price * 0.0015)
    sold_krw_price = round(selling_krw_price - commission_fee)
    sold_coin_price = convert_unit_1(selling_krw_price / readyset_quantity)
    sold_coin_quantity = convert_unit_8(readyset_quantity)

    return sold_krw_price, sold_coin_price, sold_coin_quantity, commission_fee


if __name__ == "__main__":
    UPBIT = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    # coin_names = UPBIT.get_all_coin_names()
    coin_names = ['ARK']
    start_time = dt.datetime.now()
    for i, coin_name in enumerate(coin_names):
        order_book = UPBIT.get_orderbook("KRW-" + coin_name)
        if order_book[0]['market'] == "KRW-ARK":
            print()

            bought_krw_price, bought_coin_price, bought_coin_quantity, commission_fee = get_buying_price_by_order_book(1000000, order_book[0])
            print("bought_krw_price:{0}, bought_coin_price:{1}, bought_coin_quantity:{2}, commission_fee:{3}".format(
                bought_krw_price, bought_coin_price, bought_coin_quantity, commission_fee
            ))

            sold_krw_price, sold_coin_price, sold_coin_quantity, commission_fee = get_selling_price_by_order_book(bought_coin_quantity, order_book[0])
            print("  sold_krw_price:{0},   sold_coin_price:{1},   sold_coin_quantity:{2}, commission_fee:{3}".format(
                sold_krw_price, sold_coin_price, sold_coin_quantity, commission_fee
            ))

            pp.pprint(order_book[0])
    end_time = dt.datetime.now()
    print(end_time - start_time)