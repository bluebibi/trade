import math
import sys, os
import datetime as dt

import time
from datetime import datetime, timedelta, date
import pickle
import numpy as np

from common.global_variables import CoinStatus, fmt


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
