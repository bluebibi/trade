import os, sys

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt
from upbit.upbit_api import Upbit
from predict.make_models import *

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

upbit_order_book_data = UpbitOrderBookBasedData("ADA")

x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(limit=300, split=False)

print(one_rate, total_size)

best_model = make_xgboost_model("ADA", x_normalized_original, y_up_original, total_size, one_rate)
best_model = make_gboost_model("ADA", x_normalized_original, y_up_original, total_size, one_rate)
#best_model = make_lstm_model("ADA", x_normalized_original, y_up_original, total_size, one_rate)

