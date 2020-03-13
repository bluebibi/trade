import math
import datetime as dt
import pickle
import sys, os
import glob
import subprocess

import boto3
import torch
from botocore.exceptions import ClientError

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import CoinStatus, fmt, LOCAL_MODEL_SOURCE, S3_BUCKET_NAME


def convert_unit_1(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 10) / 10
        return converted_unit
    else:
        return unit


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


def convert_unit_8(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 100000000) / 100000000
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


def elapsed_time(from_datetime, to_datetime):
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


def get_invest_krw():
    base_price = 100000
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
    return get_invest_krw()


def save_model(model, model_type):
    files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, '{0}.pkl'.format(model_type)))
    for f in files:
        os.remove(f)

    file_name = os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, '{0}.pkl'.format(model_type))
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

    s3 = boto3.client('s3')
    s3.upload_file(file_name, S3_BUCKET_NAME, '{0}.pkl'.format(model_type))

    return file_name


def load_model(model_type):
    file_name = os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, '{0}.pkl'.format(model_type))

    try:
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET_NAME, '{0}.pkl'.format(model_type), file_name)
    except ClientError as e:
        print(e, S3_BUCKET_NAME, '{0}.pkl'.format(model_type), file_name)

    files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, '{0}.pkl'.format(model_type)))
    if len(files) > 0:
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None