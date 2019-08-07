import glob
import locale

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from predict.model_cnn import CNN
from predict.model_rnn import LSTM
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from pytz import timezone
from common.utils import *
from common.logger import get_logger
from db.sqlite_handler import *

logger = get_logger("buy")
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("predict"):
    os.chdir("..")

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


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
    cnn_models = {}
    cnn_files = glob.glob(PROJECT_HOME + 'models/CNN/*.pt')

    lstm_models = {}
    lstm_files = glob.glob(PROJECT_HOME + 'models/LSTM/*.pt')

    for f in cnn_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("/home/ubuntu/git/trade/models/CNN/", "")
            model = CNN(input_width=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            cnn_models[coin_name] = model

    for f in lstm_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("/home/ubuntu/git/trade/models/LSTM/", "")
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
    current_time_str = current_time_str[:-3] + ":00"

    with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()
        all_coin_names = upbit.get_all_coin_names()
        for coin_name in all_coin_names:
            cursor.execute(select_current_base_datetime_by_datetime.format(coin_name, current_time_str))
            base_datetime_info = cursor.fetchone()
            if base_datetime_info:                  # base_datetime, ask_price_0
                coin_right_time_info[coin_name] = (base_datetime_info[0], base_datetime_info[1])
        conn.commit()

    return coin_right_time_info


def evaluate_coin_by_models(model, coin_name, model_type):
    upbit_data = UpbitOrderBookBasedData(coin_name)
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


def insert_buy_coin_info(coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, ask_price_0, buy_krw, buy_fee,
                         buy_price, buy_coin_volume, total_krw, status):
    with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
        cursor = conn.cursor()

        cursor.execute(insert_buy_try_coin_info, (
            coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, ask_price_0, buy_krw, buy_fee, buy_price,
            buy_coin_volume, total_krw, status
        ))
        conn.commit()

    msg_str = "*** BUY [{0}, ask_price_0: {1}, buy_price: {2}, buy_coin_volume: {3}, total_krw: {4}] @ {5}".format(
        coin_ticker_name,
        locale.format_string("%.2f", float(ask_price_0), grouping=True),
        locale.format_string("%.2f", float(buy_price), grouping=True),
        locale.format_string("%.2f", float(buy_coin_volume), grouping=True),
        total_krw,
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
    good_cnn_models, good_lstm_models = get_good_quality_models_for_buy()

    right_time_coin_info = get_db_right_time_coin_names()

    already_coin_ticker_names = get_coin_ticker_names_by_bought_or_trailed_status()

    target_coin_names = (set(good_cnn_models) & set(good_lstm_models) & set(right_time_coin_info)) - set(
        already_coin_ticker_names)

    logger.info("*** CNN: {0}, LSTM: {1}, Right Time Coins: {2}, Already Coins: {3}, Target Coins: {4} ***".format(
        len(good_cnn_models.keys()),
        len(good_lstm_models.keys()),
        len(right_time_coin_info.keys()),
        len(already_coin_ticker_names),
        target_coin_names
    ))

    # logger.info("*** CNN: {0}, LSTM: {1}, Right Time Coins: {2}, Already Coins: {3}, Target Coins: {4} ***".format(
    #     set(good_cnn_models.keys()),
    #     set(good_lstm_models.keys()),
    #     set(right_time_coin_info.keys()),
    #     set(already_coin_ticker_names),
    #     target_coin_names
    # ))

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
                    "ask_price_0": float(right_time_coin_info[coin_name][1]),
                    "right_time": right_time_coin_info[coin_name][0],
                    "cnn_prob": cnn_prob,
                    "lstm_prob": lstm_prob
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

                    msg_str = insert_buy_coin_info(
                        coin_ticker_name=coin_ticker_name,
                        buy_datetime=buy_try_coin_info[coin_ticker_name]['right_time'],
                        cnn_prob=buy_try_coin_info[coin_ticker_name]['cnn_prob'],
                        lstm_prob=buy_try_coin_info[coin_ticker_name]['lstm_prob'],
                        ask_price_0=buy_try_coin_info[coin_ticker_name]['ask_price_0'],
                        buy_krw=invest_krw,
                        buy_fee=buy_fee,
                        buy_price=buy_price,
                        buy_coin_volume=buy_coin_volume,
                        total_krw=current_total_krw - invest_krw,
                        status=CoinStatus.bought.value
                    )

                    SLACK.send_message("me", msg_str)
                    logger.info("{0}".format(msg_str))


if __name__ == "__main__":
    main()
