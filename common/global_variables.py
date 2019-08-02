from enum import Enum
import configparser
import torch

import sys, os

from db.sqlite_handler import SqliteHandler
from upbit.slack import PushSlack
from upbit.upbit_api import Upbit

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

class CoinStatus(Enum):
    bought = 0
    trailed = 1
    success_sold = 2
    gain_sold = 3
    loss_sold = 4


class Period(Enum):
    daily = 0
    half_daily = 1
    quater_daily = 2
    every_hour = 3


class BuyType(Enum):
    normal = 0
    prompt = 1


# GENERAL
fmt = "%Y-%m-%dT%H:%M:%S"

sqlite3_buy_sell_db_filename = os.path.join(PROJECT_HOME, 'db/upbit_buy_sell.db')
sqlite3_price_info_db_filename = os.path.join(PROJECT_HOME, 'db/upbit_price_info.db')
sqlite3_order_book_db_filename = os.path.join(PROJECT_HOME, 'db/upbit_order_book_info.db')

config = configparser.ConfigParser()
read_ok = config.read(os.getcwd()[:idx] + "upbit_auto_trade/common/config.ini")

# USER
USER_ID = int(config['USER']['user_id'])
USERNAME = config['USER']['username']
HOST_IP = config['USER']['host_ip']
SYSTEM_USERNAME = config['USER']['system_username']
SYSTEM_PASSWORD = config['USER']['system_password']
EXCHANGE = config['USER']['exchange']
SOURCE = config['USER']['source']
INITIAL_TOTAL_KRW = int(config['USER']['initial_total_krw'])
INVEST_KRW = int(config['USER']['invest_krw'])

# UPBIT
CLIENT_ID_UPBIT = config['UPBIT']['access_key']
CLIENT_SECRET_UPBIT = config['UPBIT']['secret_key']
FEE_UPBIT = 0.0005

#TELEGRAM
TELEGRAM_API_ID = config['TELEGRAM']['api_id']
TELEGRAM_API_HASH = config['TELEGRAM']['api_hash']
TELEGRAM_APP_TITLE = config['TELEGRAM']['app_title']

#SLACK
SLACK_WEBHOOK_URL_1 = config['SLACK']['webhook_url_1']
SLACK_WEBHOOK_URL_2 = config['SLACK']['webhook_url_2']

#GOOGLE
GOOGLE_APP_PASSWORD = config['GOOGLE']['app_password']

#TRAIN
NUM_EPOCHS = int(config['TRAIN']['num_epochs'])

#DATA
WINDOW_SIZE = int(config['DATA']['window_size'])
FUTURE_TARGET_SIZE = int(config['DATA']['future_target_size'])
UP_RATE = float(config['DATA']['up_rate'])

INPUT_SIZE = 125 # 1 (daily_base_timestamp) + 30 (ask_price) + 30 (ask_price_btc) + 30 (bid_price) + 30 (bid_price_btc) + 2 (total ask, total bid) + 2 (total_btc ask, total_btc bid)

VERBOSE = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPBIT = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
SLACK = PushSlack(SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2)

MIN_VALID_LOSS_THRESHOLD = float(config['EVALUATION']['min_valid_loss_threshold'])
LAST_VALID_ACCURACY_THRESHOLD = float(config['EVALUATION']['last_valid_accuracy_threshold'])
LAST_SAVE_EPOCH_THRESHOLD = int(config['EVALUATION']['last_save_epoch_threshold'])
ONE_RATE_VALID_THRESHOLD = float(config['EVALUATION']['one_rate_valid_threshold'])

#SELL
SELL_RATE = float(config['SELL']['sell_rate'])
TRANSACTION_FEE_RATE = float(config['SELL']['transaction_fee_rate'])
SELL_PERIOD = int(config['SELL']['sell_period'])

#PULL_MODELS
REMOTE_SOURCE_HOST = config['PULL_MODELS']['remote_source_host']
REMOTE_SOURCE = config['PULL_MODELS']['remote_source']
SSH_SCP_SOURCE_PORT = config['PULL_MODELS']['ssh_scp_source_port']
SSH_SCP_SOURCE_ID = config['PULL_MODELS']['ssh_scp_source_id']
SSH_SCP_SOURCE_PASSWORD = config['PULL_MODELS']['ssh_scp_source_password']
LOCAL_TARGET = config['PULL_MODELS']['local_target']

#PUSH_MODELS
IS_PUSH_AFTER_MAKE_MODELS = config.getboolean('PUSH_MODELS', 'is_push_after_make_models')
REMOTE_TARGET_HOST = config['PUSH_MODELS']['remote_target_host']
REMOTE_TARGET = config['PUSH_MODELS']['remote_target']
SSH_SCP_TARGET_PORT = config['PUSH_MODELS']['ssh_scp_target_port']
SSH_SCP_TARGET_ID = config['PUSH_MODELS']['ssh_scp_target_id']
SSH_SCP_TARGET_PEM_FILE_PATH = config['PUSH_MODELS']['ssh_scp_target_pem_file_path']
LOCAL_SOURCE = config['PUSH_MODELS']['local_source']
