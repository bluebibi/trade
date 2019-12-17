import locale
import sys, os
from pytz import timezone
import warnings
warnings.filterwarnings("ignore")

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from web.db.database import BuySell, get_order_book_class, order_book_session, buy_sell_session
from codes.predict.model_rnn import LSTM
from codes.upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from common.utils import *
from common.logger import get_logger
from codes.upbit.upbit_api import Upbit
from common.global_variables import *

logger = get_logger("buy")
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("predict"):
    os.chdir("..")

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def get_coin_ticker_names_by_bought_or_trailed_status():
    coin_ticker_name_list = []

    trades_bought_or_trailed = buy_sell_session.query(BuySell).filter(
        BuySell.status.in_([CoinStatus.bought.value, CoinStatus.trailed.value])
    ).all()

    for trade in trades_bought_or_trailed:
        coin_ticker_name_list.append(trade.coin_ticker_name.replace("KRW-", ""))

    return coin_ticker_name_list


def get_good_quality_models_for_buy():
    lstm_models = {}
    lstm_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'LSTM', '*.pt'))

    for f in lstm_files:
        if os.path.isfile(f):
            coin_name = f.split(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'LSTM'))[1].split("_")[0]
            model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            lstm_models[coin_name] = model

    return lstm_models


def get_db_right_time_coin_names():
    coin_right_time_info = {}

    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    current_time_str = current_time_str[:-4] + "0:00"

    all_coin_names = upbit.get_all_coin_names()
    for coin_name in all_coin_names:
        order_book_class = get_order_book_class(coin_name)
        q = order_book_session.query(order_book_class).filter_by(base_datetime=current_time_str)
        base_datetime_info = q.first()
        if base_datetime_info:                  # base_datetime, ask_price_0
            base_datetime = base_datetime_info.base_datetime
            ask_price_0 = base_datetime_info.ask_price_0
            coin_right_time_info[coin_name] = (base_datetime, ask_price_0)

    return coin_right_time_info


def evaluate_coin_by_model(coin_name, x, model_type="GB"):
    model = load_model(coin_name=coin_name, model_type=model_type)

    if model and x is not None:
        y_prediction = model.predict_proba(x)
        return y_prediction[0][1]
    else:
        return -1


def insert_buy_coin_info(coin_ticker_name, buy_datetime, lstm_prob, gb_prob, xgboost_prob, ask_price_0, buy_krw, buy_fee,
                         buy_price, buy_coin_volume, sell_fee, sell_krw, trail_datetime, trail_price, trail_rate, total_krw, trail_up_count, status):
    buy_sell = BuySell()
    buy_sell.coin_ticker_name = coin_ticker_name
    buy_sell.buy_datetime = buy_datetime
    buy_sell.lstm_prob = lstm_prob
    buy_sell.gb_prob = gb_prob
    buy_sell.xgboost_prob = xgboost_prob
    buy_sell.buy_base_price = ask_price_0
    buy_sell.buy_krw = buy_krw
    buy_sell.buy_fee = buy_fee
    buy_sell.buy_price = buy_price
    buy_sell.buy_coin_volume = buy_coin_volume
    buy_sell.sell_fee = sell_fee
    buy_sell.sell_krw = sell_krw
    buy_sell.trail_datetime = trail_datetime
    buy_sell.trail_price = trail_price
    buy_sell.trail_rate = trail_rate
    buy_sell.total_krw = total_krw
    buy_sell.trail_up_count = trail_up_count
    buy_sell.status = status

    buy_sell_session.add(buy_sell)
    buy_sell_session.commit()

    msg_str = "*** BUY [{0}, lstm_prob: {1}, gb_prob: {2}, xgboost_prob: {3}, ask_price_0: {4}, buy_price: {5}, buy_krw: {6}, buy_coin_volume: {7}] @ {8}".format(
        coin_ticker_name,
        locale.format_string("%.2f", float(lstm_prob), grouping=True),
        locale.format_string("%.2f", float(gb_prob), grouping=True),
        locale.format_string("%.2f", float(xgboost_prob), grouping=True),
        locale.format_string("%.2f", float(ask_price_0), grouping=True),
        locale.format_string("%.2f", float(buy_price), grouping=True),
        locale.format_string("%d", float(buy_krw), grouping=True),
        locale.format_string("%.4f", float(buy_coin_volume), grouping=True),
        SOURCE
    )

    return msg_str


def get_total_krw():
    q = buy_sell_session.query(BuySell).order_by(BuySell.id.desc()).limit(1)

    if q.first() is None:
        total_krw = INITIAL_TOTAL_KRW
    else:
        total_krw = q.first().total_krw

    return total_krw


def main():
    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    current_time_str = current_time_str[:10]

    right_time_coin_info = get_db_right_time_coin_names()
    already_coin_ticker_names = get_coin_ticker_names_by_bought_or_trailed_status()

    target_coin_names = set(right_time_coin_info) - set(already_coin_ticker_names) - set(BANNED_BUY_COIN_LIST)

    logger.info("*** Right Time Coins: {0}, Already Coins: {1}, Banned Coins: {2}, Target Coins: {3} ***".format(
        len(right_time_coin_info.keys()),
        len(already_coin_ticker_names),
        len(BANNED_BUY_COIN_LIST),
        target_coin_names
    ))

    if len(target_coin_names) > 0:
        buy_try_coin_info = {}
        buy_try_coin_ticker_names = []

        for coin_name in target_coin_names:
            upbit_data = UpbitOrderBookBasedData(coin_name)
            x = upbit_data.get_dataset_for_buy()

            gb_prob = evaluate_coin_by_model(
                coin_name=coin_name,
                x=x,
                model_type="GB"
            )

            xgboost_prob = evaluate_coin_by_model(
                coin_name=coin_name,
                x=x,
                model_type="XGBOOST"
            )

            msg_str = "{0:5} --> XGBOOST Probability:{1:7.4f}, GB Probability:{2:7.4f}".format(coin_name, xgboost_prob, gb_prob)
            if xgboost_prob > BUY_PROB_THRESHOLD and gb_prob > BUY_PROB_THRESHOLD:
                msg_str += " OK!!!"

                buy_try_coin_info["KRW-" + coin_name] = {
                    "ask_price_0": float(right_time_coin_info[coin_name][1]),
                    "right_time": right_time_coin_info[coin_name][0],
                    "lstm_prob": 0.0,
                    "gb_prob": float(gb_prob),
                    "xgboost_prob": float(xgboost_prob)
                }
                buy_try_coin_ticker_names.append("KRW-" + coin_name)

            else:
                msg_str += " - "
            logger.info(msg_str)

        if buy_try_coin_ticker_names:
            for coin_ticker_name in buy_try_coin_ticker_names:
                current_total_krw = get_total_krw()
                invest_krw = get_invest_krw_live(upbit=upbit, coin_ticker_name=coin_ticker_name)

                if True:           # if current_total_krw - invest_krw > 0:
                    _, buy_fee, buy_price, buy_coin_volume = upbit.get_expected_buy_coin_price_for_krw(
                        coin_ticker_name,
                        invest_krw,
                        TRANSACTION_FEE_RATE
                    )

                    buy_base_price = buy_try_coin_info[coin_ticker_name]['ask_price_0']

                    prompt_rising_rate = (buy_price - buy_base_price) / buy_base_price

                    if prompt_rising_rate < 0.001:
                        q = buy_sell_session.query(BuySell).filter_by(coin_ticker_name=coin_ticker_name)

                        trades_coin_ticker_name = q.all()

                        is_insert = True
                        min_buy_base_rise = sys.maxsize

                        if trades_coin_ticker_name:
                            for trade in trades_coin_ticker_name:
                                if current_time_str in trade.buy_datetime and trade.buy_base_price < min_buy_base_rise:
                                    min_buy_base_rise = trade.buy_base_price

                            logger.info("LAST CHECK: prompt_rising_rate: {0}, coin_ticker_name:{1}, min_buy_base_price:{2}, buy_price:{3}".format(
                                prompt_rising_rate,
                                coin_ticker_name,
                                min_buy_base_rise,
                                buy_price
                            ))

                            if buy_price >= min_buy_base_rise:
                                is_insert = False

                        if is_insert:
                            _, new_trail_price, sell_fee, sell_krw = upbit.get_expected_sell_coin_price_for_volume(
                                coin_ticker_name,
                                buy_coin_volume,
                                TRANSACTION_FEE_RATE
                            )

                            trail_rate = (sell_krw - invest_krw) / invest_krw

                            msg_str = insert_buy_coin_info(
                                coin_ticker_name=coin_ticker_name,
                                buy_datetime=buy_try_coin_info[coin_ticker_name]['right_time'],
                                lstm_prob=buy_try_coin_info[coin_ticker_name]['lstm_prob'],
                                gb_prob=buy_try_coin_info[coin_ticker_name]['gb_prob'],
                                xgboost_prob=buy_try_coin_info[coin_ticker_name]['xgboost_prob'],
                                ask_price_0=buy_base_price,
                                buy_krw=invest_krw,
                                buy_fee=buy_fee,
                                buy_price=buy_price,
                                buy_coin_volume=buy_coin_volume,
                                sell_fee=sell_fee,
                                sell_krw=sell_krw,
                                trail_datetime=buy_try_coin_info[coin_ticker_name]['right_time'],
                                trail_price=new_trail_price,
                                trail_rate=trail_rate,
                                total_krw=current_total_krw - invest_krw,
                                trail_up_count=0,
                                status=CoinStatus.bought.value
                            )

                            if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)
                            logger.info("{0}".format(msg_str))
                    else:
                        logger.info("coin ticker name: {0} - prompt price {1} rising is large. [buy_price: {2}, buy_base_price(ask_price_0): {3}]".format(
                            coin_ticker_name,
                            prompt_rising_rate,
                            buy_price,
                            buy_base_price
                        ))


if __name__ == "__main__":
    main()