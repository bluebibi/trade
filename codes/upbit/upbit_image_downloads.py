import os
import sys

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import requests
from codes.upbit.upbit_api import Upbit
from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt

BASE_URL = "https://static.upbit.com/logos/{0}.png"
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if __name__ == "__main__":
    coin_names = upbit.get_all_coin_names()
    for coin_name in coin_names:
        img_data = requests.get(BASE_URL.format(coin_name)).content
        with open(os.path.join(PROJECT_HOME, "web", "static", "assets", "img", "coin_images", "{0}.png".format(coin_name)), 'wb') as handler:
            handler.write(img_data)