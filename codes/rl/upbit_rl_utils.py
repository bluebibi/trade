import pickle
import sys,os

import numpy as np
import pandas as pd

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.rl.upbit_rl_constants import PERFORMANCE_FIGURE_PATH, SIZE_OF_FEATURE, SIZE_OF_FEATURE_WITHOUT_VOLUME, \
    VERBOSE_STEP, PERFORMANCE_SAVE_PATH

from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt
from common.utils import convert_unit_4, convert_unit_8, convert_unit_2

from termcolor import colored
import datetime as dt
import pprint
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


def print_before_step(env, coin_name, episode, max_episodes, num_steps, total_steps, info_dic):
    if VERBOSE_STEP:
        print_str = "[{0}:{1}/{2}:{3}/{4}, TOTAL_BALANCE: {5}, TOTAL_PROFIT: {6}, " \
                "HOLD_COIN_KRW: {7:>7d} (COIN_PRICE: {8:>8.2f}, HOLD_COINS: {9:>8.2f}), {10:>12}] ".format(
            coin_name,
            episode+1,
            max_episodes,
            num_steps+1,
            total_steps,
            env.balance + env.hold_coin_krw,
            env.total_profit,
            env.hold_coin_krw,
            env.hold_coin_unit_price,
            env.hold_coin_quantity,
            colored("TRY BUYING", "cyan") if env.status is EnvironmentStatus.TRYING_BUY else colored("TRY SELLING", "yellow"),
        )

        if env.status is EnvironmentStatus.TRYING_BUY:
            print_str += colored(" <<BUYING_COIN_UNIT_PRICE:{0:8.2f}, BUYING_COIN_QUANTITY:{1:8.2f}>>".format(
                info_dic["coin_unit_price"],
                info_dic["coin_quantity"]
            ), "cyan")
        else:
            print_str += colored(" <<SELLING_COIN_UNIT_PRICE:{0:8.2f}, SELLING_COIN_QUANTITY:{1:8.2f}>>".format(
                info_dic["coin_unit_price"],
                env.hold_coin_quantity
            ), "yellow")


def print_after_step(env, action, observation, reward, buyer_policy, seller_policy, epsilon, num_steps, episode, done):
    if VERBOSE_STEP:
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
    else:
        if (num_steps != 0 and num_steps % 1000 == 0) or done:
            header = "{0:>5}/{1}/{2}:".format(num_steps, env.total_steps, episode)

            print(header, "t_balance(profit/rate_per_sell):{0}({1}/{2:6.4f}), Buy(model):{3}/{4}={5:5.3f}({6}/{7}={8:5.3f}), "
                  "Sell(model):{9}/{10}={11:5.3f}({12}/{13}={14:5.3f}), memory_buyer/seller:{15}/{16}, Eps:{17:5.3f}".format(
                env.balance + env.hold_coin_krw, env.total_profit,
                env.total_profit_rate / env.market_sells if env.market_sells != 0 else 0.0,
                env.market_profitable_buys, env.market_buys,
                env.market_profitable_buys / env.market_buys if env.market_buys != 0 else 0.0,
                env.market_profitable_buys_from_model, env.market_buys_from_model,
                env.market_profitable_buys_from_model / env.market_buys_from_model if env.market_buys_from_model != 0 else 0.0,
                env.market_profitable_sells, env.market_sells,
                env.market_profitable_sells / env.market_sells if env.market_sells != 0 else 0.0,
                env.market_profitable_sells_from_model, env.market_sells_from_model,
                env.market_profitable_sells_from_model / env.market_sells_from_model if env.market_sells_from_model != 0 else 0.0,
                buyer_policy.buyer_memory.size(), seller_policy.seller_memory.size(),
                epsilon * 100
            ), end="\n")


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


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        if len(values) == 0:
            sma = 0.0
        else:
            sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


def draw_performance(env, args):
    if os.path.exists(PERFORMANCE_FIGURE_PATH):
        os.remove(PERFORMANCE_FIGURE_PATH)

    plt.figure(figsize=(18, 18))
    title_str = "DQN "

    if args.per:
        title_str += "[Prioritized Experience Memory] "
    else:
        title_str += "[Experience Memory] "

    if args.volume:
        title_str += "[WITH VOLUME FEATURES (FEATURE SIZE:{0})] ".format(SIZE_OF_FEATURE)
    else:
        title_str += "[WITHOUT VOLUME FEATURES (FEATURE SIZE:{0})] ".format(SIZE_OF_FEATURE_WITHOUT_VOLUME)

    if args.lstm:
        title_str += "[LSTM] "
    else:
        title_str += "[CNN] "

    plt.suptitle(title_str, fontsize=14)

    plt.subplot(421)
    plt.yscale('log', basey=2)
    plt.plot(range(1, len(env.market_buy_list) + 1), env.market_buy_list, label="Buys")
    plt.plot(range(1, len(env.market_buy_by_model_list) + 1), env.market_buy_by_model_list, linestyle="--", label="Buys by model")
    plt.plot(range(1, len(env.market_profitable_buy_list) + 1), env.market_profitable_buy_list, linestyle="-.", label="Profitable buys")
    plt.plot(range(1, len(env.market_profitable_buy_by_model_list) + 1), env.market_profitable_buy_by_model_list, linestyle=":", label="Profitable buys by model")
    plt.title('MARKET BUYS', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(422)
    plt.yscale('log', basey=2)
    plt.plot(range(1, len(env.market_sell_list) + 1), env.market_sell_list, label="Sells")
    plt.plot(range(1, len(env.market_sell_by_model_list) + 1), env.market_sell_by_model_list, linestyle="--", label="Sells by model")
    plt.plot(range(1, len(env.market_profitable_sell_list) + 1), env.market_profitable_sell_list, linestyle="-.", label="Profitable sells")
    plt.plot(range(1, len(env.market_profitable_sell_by_model_list) + 1), env.market_profitable_sell_by_model_list, linestyle=":", label="Profitable sells by model")
    plt.title('MARKET SELLS', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(423)
    plt.plot(range(1, len(env.buyer_loss_list) + 1), env.buyer_loss_list)
    plt.title('BUYER LOSS', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.grid()

    plt.subplot(424)
    plt.plot(range(1, len(env.seller_loss_list) + 1), env.seller_loss_list)
    plt.title('SELLER LOSS', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.grid()

    plt.subplot(413)
    plt.plot(range(1, len(env.score_list) + 1), env.score_list)
    plt.plot(
        range(len(env.score_list)),
        exp_moving_average(env.score_list, 10),
        color="red"
    )
    plt.title('SCORE', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.grid()

    plt.subplot(414)
    plt.plot(range(1, len(env.total_balance_per_episode_list) + 1), env.total_balance_per_episode_list)
    plt.plot(
        range(len(env.total_balance_per_episode_list)),
        exp_moving_average(env.total_balance_per_episode_list, 10),
        color="red"
    )
    plt.title('TOTAL_BALANCE_PER_EPISODE', fontweight="bold", size=10)
    plt.xlabel('EPISODES', size=10)
    plt.grid()

    plt.savefig(PERFORMANCE_FIGURE_PATH)

    performance_dic = {
        "market_buy_list": env.market_buy_list,
        "market_buy_by_model_list": env.market_buy_by_model_list,
        "market_profitable_buy_list": env.market_profitable_buy_list,
        "market_profitable_buy_by_model_list": env.market_profitable_buy_by_model_list,
        "market_sell_list": env.market_sell_list,
        "market_sell_by_model_list": env.market_sell_by_model_list,
        "market_profitable_sell_list": env.market_profitable_sell_list,
        "market_profitable_sell_by_model_list": env.market_profitable_sell_by_model_list,
        "buyer_loss_list": env.buyer_loss_list,
        "seller_loss_list": env.seller_loss_list,
        "score_list": env.score_list,
        "total_balance_per_episode_list": env.total_balance_per_episode_list
    }

    with open(os.path.join(PERFORMANCE_SAVE_PATH, 'performance.pkl'), 'wb') as fout:
        pickle.dump(performance_dic, fout)


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