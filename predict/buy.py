import glob
import locale
import sys, os
from pytz import timezone
import warnings
warnings.filterwarnings("ignore")

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from predict.model_rnn import LSTM
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from common.utils import *
from common.logger import get_logger
from db.sqlite_handler import *
from common.utils import load_gb_model

logger = get_logger("buy")
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("predict"):
    os.chdir("..")

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

if SELF_MODELS_MODE:
    model_source = SELF_MODEL_SOURCE
else:
    model_source = LOCAL_MODEL_SOURCE


def get_coin_ticker_names_by_bought_or_trailed_status():
    coin_ticker_name_list = []
    with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_coin_ticker_name_by_status_sql)

        for coin_ticker_name in rows:
            coin_ticker_name_list.append(coin_ticker_name[0].replace("KRW-", ""))
        conn.commit()
    return coin_ticker_name_list


def get_good_quality_models_for_buy():
    lstm_models = {}
    lstm_files = glob.glob(PROJECT_HOME + '{0}LSTM/*.pt'.format(model_source))

    for f in lstm_files:
        if os.path.isfile(f):
            coin_name = f.split(PROJECT_HOME + '{0}LSTM/'.format(model_source))[1].split("_")[0]
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

    with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()
        all_coin_names = upbit.get_all_coin_names()
        for coin_name in all_coin_names:
            cursor.execute(select_current_base_datetime_by_datetime.format(coin_name, current_time_str))
            base_datetime_info = cursor.fetchone()
            if base_datetime_info:                  # base_datetime, ask_price_0
                base_datetime = base_datetime_info[0]
                ask_price_0 = base_datetime_info[1]
                coin_right_time_info[coin_name] = (base_datetime, ask_price_0)
        conn.commit()

    return coin_right_time_info


def evaluate_coin_by_gb_model(coin_name, x):
    model = load_gb_model(coin_name=coin_name)

    if model:
        y_prediction = model.predict_proba(x)
        return y_prediction[0][1]
    else:
        return -1


def evaluate_coin_by_lstm_model(coin_name, x):
    model = load_lstm_model(coin_name=coin_name)

    if model:
        y_prediction = model.predict_proba(x)
        return y_prediction[0][1]
    else:
        return -1


def insert_buy_coin_info(coin_ticker_name, buy_datetime, lstm_prob, gb_prob, ask_price_0, buy_krw, buy_fee,
                         buy_price, buy_coin_volume, total_krw, status):
    with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()

        cursor.execute(insert_buy_try_coin_info, (
            coin_ticker_name, buy_datetime,
            lstm_prob, gb_prob, ask_price_0,
            buy_krw, buy_fee,
            buy_price, buy_coin_volume,
            total_krw, status
        ))
        conn.commit()

    msg_str = "*** BUY [{0}, lstm_prob: {1}, gb_prob: {2}, ask_price_0: {3}, buy_price: {4}, buy_krw: {5}, buy_coin_volume: {6}] @ {7}".format(
        coin_ticker_name,
        locale.format_string("%.2f", float(lstm_prob), grouping=True),
        locale.format_string("%.2f", float(gb_prob), grouping=True),
        locale.format_string("%.2f", float(ask_price_0), grouping=True),
        locale.format_string("%.2f", float(buy_price), grouping=True),
        locale.format_string("%d", float(buy_krw), grouping=True),
        locale.format_string("%.4f", float(buy_coin_volume), grouping=True),
        SOURCE
    )

    return msg_str


def get_total_krw():
    with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(select_total_krw)

        total_krw_ = cursor.fetchone()
        if total_krw_:
            total_krw = int(total_krw_[0])
        else:
            total_krw = INITIAL_TOTAL_KRW

    return total_krw


def main():
    good_lstm_models = get_good_quality_models_for_buy()

    right_time_coin_info = get_db_right_time_coin_names()

    already_coin_ticker_names = get_coin_ticker_names_by_bought_or_trailed_status()

    target_coin_names = (set(good_lstm_models) & set(right_time_coin_info)) - set(already_coin_ticker_names) - set(BANNED_BUY_COIN_LIST)

    logger.info("*** LSTM: {0}, Right Time Coins: {1}, Already Coins: {2}, Target Coins: {3} ***".format(
        len(good_lstm_models.keys()),
        len(right_time_coin_info.keys()),
        len(already_coin_ticker_names),
        target_coin_names
    ))

    # logger.info("*** LSTM: {0}, Right Time Coins: {1}, Already Coins: {2}, Target Coins: {3} ***".format(
    #     set(good_lstm_models.keys()),
    #     set(right_time_coin_info.keys()),
    #     set(already_coin_ticker_names),
    #     target_coin_names
    # ))

    if len(target_coin_names) > 0:
        buy_try_coin_info = {}
        buy_try_coin_ticker_names = []

        for coin_name in target_coin_names:
            upbit_data = UpbitOrderBookBasedData(coin_name)
            x = upbit_data.get_dataset_for_buy(model_type="GB")

            lstm_prob = evaluate_coin_by_lstm_model(
                coin_name=coin_name,
                x=x
            )

            gb_prob = evaluate_coin_by_gb_model(
                coin_name=coin_name,
                x=x
            )

            msg_str = "{0:5} --> LSTM Probability:{1:7.4f}, GB Probability:{2:7.4f}".format(coin_name, lstm_prob, gb_prob)
            if lstm_prob > LSTM_BUY_PROB_THRESHOLD and gb_prob > GRADIENT_BOOSTING_BUY_PROB_THRESHOLD:
                msg_str += " OK!!!"
            else:
                msg_str += " - "
            logger.info(msg_str)

            if lstm_prob > LSTM_BUY_PROB_THRESHOLD and gb_prob > GRADIENT_BOOSTING_BUY_PROB_THRESHOLD:
                # coin_name --> right_time, prob
                buy_try_coin_info["KRW-" + coin_name] = {
                    "ask_price_0": float(right_time_coin_info[coin_name][1]),
                    "right_time": right_time_coin_info[coin_name][0],
                    "lstm_prob": lstm_prob,
                    "gb_prob": gb_prob
                }
                buy_try_coin_ticker_names.append("KRW-" + coin_name)

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
                        with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
                            cursor = conn.cursor()
                            cursor.execute(select_buy_prohibited_coins_sql, (
                                coin_ticker_name,
                                buy_try_coin_info[coin_ticker_name]['right_time']
                            ))

                            rows = cursor.fetchall()

                            is_insert = False
                            min_buy_base_rise = sys.maxsize
                            if rows:
                                for row in rows:
                                    if float(row[0]) < min_buy_base_rise:
                                        min_buy_base_rise = float(row[0])

                                logger.info("LAST CHECK: prompt_rising_rate: {0}, coin_ticker_name:{1}, min_buy_base_price:{2}, buy_price:{3}".format(
                                    prompt_rising_rate,
                                    coin_ticker_name,
                                    min_buy_base_rise,
                                    buy_price
                                ))

                                if buy_price < min_buy_base_rise:
                                    is_insert = True
                            else:
                                is_insert = True

                            if is_insert:
                                msg_str = insert_buy_coin_info(
                                    coin_ticker_name=coin_ticker_name,
                                    buy_datetime=buy_try_coin_info[coin_ticker_name]['right_time'],
                                    lstm_prob=buy_try_coin_info[coin_ticker_name]['lstm_prob'],
                                    gb_prob=buy_try_coin_info[coin_ticker_name]['gb_prob'],
                                    ask_price_0=buy_base_price,
                                    buy_krw=invest_krw,
                                    buy_fee=buy_fee,
                                    buy_price=buy_price,
                                    buy_coin_volume=buy_coin_volume,
                                    total_krw=current_total_krw - invest_krw,
                                    status=CoinStatus.bought.value
                                )

                                if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)
                                logger.info("{0}".format(msg_str))
                    else:
                        logger.info("coin ticker name: {0} - prompt price {1} rising is large".format(coin_ticker_name, prompt_rising_rate))


if __name__ == "__main__":
    main()
