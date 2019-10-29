import math
import datetime as dt
import pickle
import sys, os
import glob
import subprocess
import torch

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import CoinStatus, fmt
from common.global_variables import SELF_MODELS_MODE, SELF_MODEL_SOURCE, LOCAL_MODEL_SOURCE

if SELF_MODELS_MODE:
    model_source = SELF_MODEL_SOURCE
else:
    model_source = LOCAL_MODEL_SOURCE


def convert_unit_2(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 100) / 100
        return converted_unit
    else:
        return unit


def convert_unit_4(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 10000) / 10000
        return converted_unit
    else:
        return unit


def coin_status_to_hangul(status):
    if status == CoinStatus.bought.value:
        status = "구매"
    elif status == CoinStatus.trailed.value:
        status = "추적"
    elif status == CoinStatus.success_sold.value:
        status = "성공"
    elif status == CoinStatus.gain_sold.value:
        status = "이득"
    elif status == CoinStatus.loss_sold.value:
        status = "손실"
    elif status == CoinStatus.up_trailed.value:
        status = "이득추적"
    return status


def elapsed_time_str(from_datetime, to_datetime):
    from_datetime = dt.datetime.strptime(from_datetime, fmt.replace("T", " "))
    to_datetime = dt.datetime.strptime(to_datetime, fmt.replace("T", " "))
    time_diff = to_datetime - from_datetime
    time_diff_hours = int(time_diff.seconds / 3600)
    time_diff_minutes = int((time_diff.seconds - time_diff_hours * 3600) / 60)
    return "{:0>2d}:{:0>2d}".format(time_diff_hours, time_diff_minutes)


def convert_to_daily_timestamp(datetime_str):
    time_str_hour = datetime_str.split(" ")[1].split(":")[0]
    time_str_minute = datetime_str.split(" ")[1].split(":")[1]

    if time_str_hour[0] == "0":
        time_str_hour = time_str_hour[1:]

    if time_str_minute[0] == "0":
        time_str_minute = time_str_minute[1:]

    daily_base_timestamp = int(time_str_hour) * 100 + int(time_str_minute)

    return daily_base_timestamp


def get_invest_krw(current_price, total_ask_size, total_bid_size):
    base_price = current_price * (total_ask_size + total_bid_size) * 0.001
    if base_price > 300000:
        return 300000
    elif 150000 < base_price <= 300000:
        return 200000
    else:
        return 100000


def get_invest_krw_live(upbit, coin_ticker_name):
    info = upbit.get_orderbook(tickers=coin_ticker_name)
    current_price = info[0]['orderbook_units'][0]['ask_price']
    total_ask_size = info[0]['total_ask_size']
    total_bid_size = info[0]['total_bid_size']
    return get_invest_krw(current_price, total_ask_size, total_bid_size)


def save_model(coin_name, model, model_type="GB"):
    files = glob.glob(PROJECT_HOME + '{0}{1}/{2}.pkl'.format(model_source, model_type, coin_name))
    for f in files:
        os.remove(f)

    file_name = "{0}{1}/{2}.pkl".format(model_source, model_type, coin_name)
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(coin_name, model_type="GB"):
    files = glob.glob(PROJECT_HOME + '{0}{1}/{2}.pkl'.format(model_source, model_type, coin_name))
    if len(files) > 0:
        file_name = "{0}{1}/{2}.pkl".format(model_source, model_type, coin_name)
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None