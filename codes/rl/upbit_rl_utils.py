import sys,os

from codes.rl.upbit_rl_constants import PERFORMANCE_FIGURE_PATH
from db.database import naver_order_book_session, get_order_book_class

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt, WINDOW_SIZE
from common.utils import convert_unit_4, convert_unit_8, convert_unit_2

from termcolor import colored
import datetime as dt
import pprint
import numpy as np
import pandas as pd
import enum
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=2)


class BuyerAction(enum.Enum):
    BUY_HOLD = 0,
    MARKET_BUY = 1


class SellerAction(enum.Enum):
    SELL_HOLD = 0,
    MARKET_SELL = 1


class EnvironmentType(enum.Enum):
    TRAIN_VALID = 0
    LIVE = 1


class EnvironmentStatus(enum.Enum):
    TRYING_BUY = 0
    TRYING_SELL = 1

features = ["daily_base_timestamp",
        "ask_price_0", "ask_size_0", "ask_price_1", "ask_size_1", "ask_price_2", "ask_size_2", "ask_price_3",
        "ask_size_3", "ask_price_4", "ask_size_4",
        "bid_price_0", "bid_size_0", "bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2", "bid_price_3",
        "bid_size_3", "bid_price_4", "bid_size_4"]

def get_rl_dataset(coin_name):
    order_book_class = get_order_book_class(coin_name)
    queryset = naver_order_book_session.query(order_book_class).order_by(order_book_class.base_datetime.asc())
    df = pd.read_sql(queryset.statement, naver_order_book_session.bind)

    # df = df.drop(["id", "base_datetime", "collect_timestamp"], axis=1)
    base_datetime_df = df.filter(["base_datetime"], axis=1)

    df = df.filter(features, axis=1)

    for feature in features:
        df[feature].mask(df[feature] == 0.0, 0.1, inplace=True)

    base_datetime_data = pd.to_datetime(base_datetime_df["base_datetime"])
    data = df.values

    dim_0 = data.shape[0] - WINDOW_SIZE + 1
    dim_1 = data.shape[1]

    base_datetime_X = []
    X = np.zeros(shape=(dim_0, WINDOW_SIZE, dim_1))

    for i in range(dim_0):
        X[i] = data[i: i + WINDOW_SIZE]
        base_datetime_X.append(str(base_datetime_data[i + WINDOW_SIZE - 1]))

    base_datetime_X = np.asarray(base_datetime_X)

    total_size = X.shape[0]

    indices = list(range(total_size))
    train_indices = list(set(indices[:int(total_size * 0.8)]))
    valid_indices = list(set(range(total_size)) - set(train_indices))
    x_train = X[train_indices]
    x_train_base_datetime = base_datetime_X[train_indices]
    x_valid = X[valid_indices]
    x_valid_base_datetime = base_datetime_X[valid_indices]

    train_size = x_train.shape[0]
    valid_size = x_valid.shape[0]

    return x_train, x_train_base_datetime, train_size, x_valid, x_valid_base_datetime, valid_size

def print_before_step(env, coin_name, episode, max_episodes, num_steps, info_dic):
    if num_steps == 0:
        print("[COIN_NAME: {0}] EPISODES & STEPS".format(coin_name))

    print_str = "[{0}/{1:>2d}/{2:>4d}, Balance: {3:>7}, TOTAL_PROFIT: {4:>6}, " \
                "HOLD_COIN_KRW: {5:>7d} (COIN_PRICE: {6:>8.2f}, HOLD_COINS: {7:>8.2f}), {8:>12}] ".format(
        max_episodes,
        episode,
        num_steps,
        env.balance,
        env.total_profit,
        env.hold_coin_krw,
        env.hold_coin_unit_price,
        env.hold_coin_quantity,
        colored("TRY BUYING", "cyan") if env.status is EnvironmentStatus.TRYING_BUY else colored("TRY SELLING", "yellow"),
    )

    if env.status is EnvironmentStatus.TRYING_BUY:
        print_str += colored(" <<change_index:{0:8.2f}, buying_coin_unit_price:{1:8.2f}, buying_coin_quantity:{2:8.2f}>>".format(
            info_dic["change_index"],
            info_dic["coin_unit_price"],
            info_dic["coin_quantity"]
        ), "cyan")
    else:
        print_str += colored(" <<change_index:{0:8.2f}, selling_coin_unit_price:{1:8.2f}, selling_coin_quantity:{2:8.2f}>>".format(
            info_dic["change_index"],
            info_dic["coin_unit_price"],
            env.hold_coin_quantity
        ), "yellow")

    print(print_str, end=" ")


def print_after_step(env, action, observation, reward, done, buyer_policy, seller_policy, epsilon):
    if env.status is EnvironmentStatus.TRYING_BUY:
        if action is 0 or action is BuyerAction.BUY_HOLD:
            action_str = colored("  BUY HOLD [HOLD_COIN_KRW: {0:7d} (COIN_PRICE: {1:>8.2f}, HOLD_COINS: {2:>8.2f})]".format(
                env.hold_coin_krw, env.hold_coin_unit_price, env.hold_coin_quantity
            ), "magenta")
        elif action is 1 or action is BuyerAction.MARKET_BUY:
            action_str = colored(" MARKET_BUY [BOUGHT_COIN_KRW: {0:7d} (COIN_PRICE: {1:>8.2f}, BOUGHT_COINS: {2:>8.2f})]".format(
                env.just_bought_coin_krw, env.just_bought_coin_unit_price, env.just_bought_coin_quantity
            ), "red")
        else:
            raise ValueError("print_after_step: action value: {0}".format(action))
    else:
        if action is 0 or action is SellerAction.SELL_HOLD:
            action_str = colored("  SELL_HOLD [HOLD_COIN_KRW: {0:7d} (COIN_PRICE: {1:>8.2f}, HOLD_COINS: {2:>8.2f})]".format(
                env.hold_coin_krw, env.hold_coin_unit_price, env.hold_coin_quantity
            ), "green")
        elif action is 1 or action is SellerAction.MARKET_SELL:
            action_str = colored("MARKET_SELL [SOLD_COIN_KRW: {0:7d} (COIN_PRICE: {1:>8.2f}, SOLD_COINS: {2:>8.2f}) || PROFIT: {3}]".format(
                env.just_sold_coin_krw, env.just_sold_coin_unit_price, env.just_sold_coin_quantity, env.just_sold_coin_krw - env.just_bought_coin_krw
            ), "blue")
        else:
            raise ValueError("print_after_step: action value: {0}".format(action))

    print_str = "\n==> ACTION: {0:}, OBSERVATION:{1}, REWARD: {2:7}, BUYER_MEMORY: {3:6} (PENDING: {4:1}), SELLER_MEMORY: {5:6}, EPSILON: {6:4.3f}%".format(
        action_str, observation.shape, reward,
        buyer_policy.buyer_memory.size(),
        1 if buyer_policy.pending_buyer_transition else 0,
        seller_policy.seller_memory.size(),
        epsilon * 100
    )
    print(print_str, end="\n\n")


def array_2d_to_dict_list_order_book(arr_data):
    order_book = dict()
    order_book['orderbook_units'] = []
    for idx in range(5):
        item = {}
        item['ask_price'] = arr_data[1 + idx * 2]
        item['ask_size'] = arr_data[2 + idx * 2]
        item['bid_price'] = arr_data[11 + idx * 2]
        item['bid_size'] = arr_data[12 + idx * 2]
        order_book['orderbook_units'].append(item)
    return order_book


def get_buying_price_by_order_book(readyset_krw, order_book):
    commission_fee = int(readyset_krw * 0.0015)
    buying_krw = readyset_krw - commission_fee
    bought_coin_quantity = 0.0
    for order in order_book['orderbook_units']:
        units_ask = convert_unit_4(buying_krw / float(order['ask_price']))
        if units_ask <= float(order['ask_size']):
            bought_coin_quantity += units_ask
            buying_krw = buying_krw - float(order['ask_price']) * units_ask
            break
        else:
            buying_krw = buying_krw - float(order['ask_price']) * float(order['ask_size'])
            bought_coin_quantity += float(order['ask_size'])

    bought_coin_quantity = convert_unit_8(bought_coin_quantity)
    bought_coin_krw = int(round(readyset_krw - buying_krw - commission_fee))
    bought_coin_unit_price = convert_unit_2(bought_coin_krw / bought_coin_quantity)

    return bought_coin_krw, bought_coin_unit_price, bought_coin_quantity, commission_fee


def get_selling_price_by_order_book(readyset_quantity, order_book):
    selling_quantity = readyset_quantity
    selling_krw = 0.0
    for order in order_book['orderbook_units']:
        if selling_quantity <= float(order['bid_size']):
            selling_krw += order['bid_price'] * selling_quantity
            break
        else:
            selling_quantity = selling_quantity - float(order['bid_size'])
            selling_krw += order['bid_price'] * float(order['bid_size'])

    commission_fee = int(round(selling_krw * 0.0015))
    sold_coin_krw = int(round(selling_krw - commission_fee))
    sold_coin_unit_price = convert_unit_2(selling_krw / readyset_quantity)
    sold_coin_quantity = convert_unit_8(readyset_quantity)

    return sold_coin_krw, sold_coin_unit_price, sold_coin_quantity, commission_fee

def draw_performance(total_profit_list, buyer_loss_list, seller_loss_list, market_buy_list, market_sell_list,
                     market_buy_from_model_list, market_sell_from_model_list):
    if os.path.exists(PERFORMANCE_FIGURE_PATH):
        os.remove(PERFORMANCE_FIGURE_PATH)

    plt.clf()

    plt.figure(figsize=(20, 10))

    plt.subplot(321)
    plt.plot(range(len(market_buy_list)), market_buy_list)
    plt.plot(range(len(market_buy_from_model_list)), market_buy_from_model_list, linestyle="--")
    plt.title('MARKET BUYS', fontweight="bold", size=10)
    plt.grid()

    plt.subplot(322)
    plt.plot(range(len(market_sell_list)), market_sell_list)
    plt.plot(range(len(market_sell_from_model_list)), market_sell_from_model_list, linestyle="--")
    plt.title('MARKET SELLS', fontweight="bold", size=10)
    plt.grid()

    plt.subplot(323)
    plt.plot(range(len(buyer_loss_list)), buyer_loss_list)
    plt.title('BUYER LOSS', fontweight="bold", size=10)
    plt.grid()

    plt.subplot(324)
    plt.plot(range(len(seller_loss_list)), seller_loss_list)
    plt.title('SELLER LOSS', fontweight="bold", size=10)
    plt.grid()

    plt.subplot(313)
    plt.plot(range(len(total_profit_list)), total_profit_list)
    plt.title('TOTAL_PROFIT', fontweight="bold", size=10)
    plt.xlabel('STEPS', size=10)
    plt.grid()

    plt.savefig(PERFORMANCE_FIGURE_PATH)
    plt.close('all')


if __name__ == "__main__":
    UPBIT = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    # coin_names = UPBIT.get_all_coin_names()
    coin_names = ['ARK']
    start_time = dt.datetime.now()
    for i, coin_name in enumerate(coin_names):
        order_book = UPBIT.get_orderbook("KRW-" + coin_name)
        if order_book[0]['market'] == "KRW-ARK":
            print()

            bought_coin_krw, bought_coin_unit_price, bought_coin_quantity, commission_fee = get_buying_price_by_order_book(1000000, order_book[0])
            print("bought_coin_krw:{0}, bought_coin_unit_price:{1}, bought_coin_quantity:{2}, commission_fee:{3}".format(
                bought_coin_krw, bought_coin_unit_price, bought_coin_quantity, commission_fee
            ))

            sold_coin_krw, sold_coin_unit_price, sold_coin_quantity, commission_fee = get_selling_price_by_order_book(bought_coin_quantity, order_book[0])
            print("  sold_coin_krw:{0},   sold_coin_unit_price:{1},   sold_coin_quantity:{2}, commission_fee:{3}".format(
                sold_coin_krw, sold_coin_unit_price, sold_coin_quantity, commission_fee
            ))

            pp.pprint(order_book[0])
    end_time = dt.datetime.now()
    print(end_time - start_time)