import glob
import locale
import sqlite3

import sys, os
idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from predict.model_cnn import CNN
from predict.model_rnn import LSTM
from upbit.upbit_data import UpbitData
from pytz import timezone
from common.utils import *
from common.logger import get_logger

logger = get_logger("buy_logger")

if os.getcwd().endswith("predict"):
    os.chdir("..")


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

select_coin_ticker_name_by_status_sql = "SELECT coin_ticker_name FROM BUY_SELL WHERE status=0 or status=1;"
select_by_datetime = "SELECT close_price FROM {0} WHERE datetime='{1}';"
select_total_krw = "SELECT total_krw FROM BUY_SELL ORDER BY id DESC LIMIT 1;"
insert_buy_try_coin_info = "INSERT INTO BUY_SELL (coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, buy_base_price, " \
                           "buy_krw, buy_fee, buy_price, buy_coin_volume, total_krw, status) VALUES (?, ?, ?, ?, ?, " \
                           "?, ?, ?, ?, ?, ?);"


def get_coin_ticker_name_by_status():
    coin_ticker_name_list = []
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_coin_ticker_name_by_status_sql)

        for coin_ticker_name in rows:
            coin_ticker_name_list.append(coin_ticker_name[0].replace("KRW-", ""))
        conn.commit()
    return coin_ticker_name_list


def get_good_quality_coin_names_for_buy():
    cnn_models = {}
    cnn_files = glob.glob('./models/CNN/*.pt')

    lstm_models = {}
    lstm_files = glob.glob('./models/LSTM/*.pt')

    for f in cnn_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("./models/CNN/", "")
            model = CNN(input_width=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            cnn_models[coin_name] = model

    for f in lstm_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("./models/LSTM/", "")
            model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            lstm_models[coin_name] = model

    return cnn_models, lstm_models


def get_db_right_time_coin_names():
    coin_right_time_info = {}
    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    current_time_str = current_time_str[:-4] + "0:00"

    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()
        all_coin_names = UPBIT.get_all_coin_names()
        for coin_name in all_coin_names:
            cursor.execute(select_by_datetime.format("KRW_" + coin_name, current_time_str))
            close_price_ = cursor.fetchall()
            if close_price_:
                coin_right_time_info[coin_name] = [close_price_[0][0], current_time_str]
        conn.commit()

    return coin_right_time_info


def evaluate_coin_by_models(model, coin_name, model_type):
    upbit_data = UpbitData(coin_name)
    x = upbit_data.get_buy_for_data(model_type=model_type)

    out = model.forward(x)
    out = torch.sigmoid(out)
    t = torch.Tensor([0.5]).to(DEVICE)
    output_index = (out > t).float() * 1

    prob = out.item()
    idx = int(output_index.item())

    if idx and prob > 0.9:
        return prob
    else:
        return -1


def insert_buy_coin_info(coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, buy_base_price, buy_krw, buy_fee,
                         buy_price, buy_coin_volume, total_krw, status):
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        "INSERT INTO BUY_SELL (coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, buy_base_price, " \
        "buy_krw, buy_fee, buy_price, buy_coin_volume, total_krw, status) VALUES (?, ?, ?, ?, ?, " \
        "?, ?, ?, ?, ?, ?);"

        cursor.execute(insert_buy_try_coin_info, (
            coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, buy_base_price, buy_krw, buy_fee, buy_price,
            buy_coin_volume, total_krw, status
        ))
        conn.commit()

    msg_str = "*** BUY [{0}, buy_base_price: {1}, buy_price: {2}, buy_coin_volume: {3}, total_krw: {4}] @ {5}".format(
        coin_ticker_name,
        locale.format_string("%.2f", float(buy_base_price), grouping=True),
        locale.format_string("%.2f", float(buy_price), grouping=True),
        locale.format_string("%.2f", float(buy_coin_volume), grouping=True),
        total_krw,
        SOURCE
    )

    return msg_str


def get_total_krw():
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None,
                         check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(select_total_krw)

        total_krw_ = cursor.fetchone()
        if total_krw_:
            total_krw = int(total_krw_[0])
        else:
            total_krw = INITIAL_TOTAL_KRW

    return total_krw


def main():
    good_cnn_models, good_lstm_models = get_good_quality_coin_names_for_buy()

    right_time_coin_info = get_db_right_time_coin_names()

    already_coin_ticker_names = get_coin_ticker_name_by_status()

    target_coin_names = (set(good_cnn_models) & set(good_lstm_models) & set(right_time_coin_info)) - set(
        already_coin_ticker_names)

    logger.info("*** CNN: {0}, LSTM: {1}, Right Time Coins: {2}, Target Coins: {3} ***".format(
        len(good_cnn_models),
        len(good_lstm_models),
        len(right_time_coin_info),
        target_coin_names
    ))

    if len(target_coin_names) > 0:
        buy_try_coin_info = {}
        buy_try_coin_ticker_names = []

        for coin_name in target_coin_names:
            cnn_prob = evaluate_coin_by_models(
                model=good_cnn_models[coin_name],
                coin_name=coin_name,
                model_type="CNN"
            )

            lstm_prob = evaluate_coin_by_models(
                model=good_lstm_models[coin_name],
                coin_name=coin_name,
                model_type="LSTM"
            )

            logger.info("{0} --> CNN Probability:{1:.4f}, LSTM Probability:{2:.4f}".format(
                coin_name, cnn_prob, lstm_prob
            ))

            if cnn_prob > 0 and lstm_prob > 0:
                # coin_name --> right_time, prob
                buy_try_coin_info["KRW-" + coin_name] = {
                    "buy_base_price": float(right_time_coin_info[coin_name][0]),
                    "right_time": right_time_coin_info[coin_name][1],
                    "cnn_prob": cnn_prob,
                    "lstm_prob": lstm_prob
                }
                buy_try_coin_ticker_names.append("KRW-" + coin_name)

        if buy_try_coin_ticker_names:
            for coin_ticker_name in buy_try_coin_ticker_names:
                current_total_krw = get_total_krw()
                if True:
#                if current_total_krw - INVEST_KRW > 0:

                    _, buy_fee, buy_price, buy_coin_volume = UPBIT.get_expected_buy_coin_price_for_krw(
                        coin_ticker_name,
                        INVEST_KRW,
                        TRANSACTION_FEE_RATE
                    )

                    msg_str = insert_buy_coin_info(
                        coin_ticker_name=coin_ticker_name,
                        buy_datetime=buy_try_coin_info[coin_ticker_name]['right_time'],
                        cnn_prob=buy_try_coin_info[coin_ticker_name]['cnn_prob'],
                        lstm_prob=buy_try_coin_info[coin_ticker_name]['lstm_prob'],
                        buy_base_price=buy_try_coin_info[coin_ticker_name]['buy_base_price'],
                        buy_krw=INVEST_KRW,
                        buy_fee=buy_fee,
                        buy_price=buy_price,
                        buy_coin_volume=buy_coin_volume,
                        total_krw=current_total_krw - INVEST_KRW,
                        status=CoinStatus.bought.value
                    )

                    SLACK.send_message("me", msg_str)
                    logger.info("{0}".format(msg_str))


if __name__ == "__main__":
    main()
