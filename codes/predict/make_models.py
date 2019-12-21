# https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn
import time
import sys, os

from sklearn.metrics import f1_score

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import xgboost as xgb
from pytz import timezone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from common.global_variables import *
import matplotlib.pyplot as plt
from common.utils import *
from web.db.database import Model, trade_db_session

import numpy as np
import os
from common.logger import get_logger
from codes.upbit.upbit_api import Upbit
from codes.upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from common.utils import save_model

import torch.nn.modules.loss
import warnings
import gc
warnings.filterwarnings("ignore")

logger = get_logger("make_models")

def mkdir_models():
    if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE)):
        os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE))

    if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'graphs')):
        os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'graphs'))

    if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'SCALERS')):
        os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'SCALERS'))


def save_graph(coin_name, model_type, valid_loss_min, last_valid_accuracy, last_save_epoch, valid_size, one_count_rate,
               avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list):
    files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type, coin_name + '*'))
    for f in files:
        os.remove(f)

    plt.clf()

    fig, ax_lst = plt.subplots(2, 2, figsize=(30, 10), gridspec_kw={'hspace': 0.35})

    ax_lst[0][0].plot(range(len(avg_train_losses)), avg_train_losses)
    ax_lst[0][0].set_title('AVG. TRAIN LOSSES', fontweight="bold", size=10)

    ax_lst[0][1].plot(range(len(train_accuracy_list)), train_accuracy_list)
    ax_lst[0][1].set_title('TRAIN ACCURACY CHANGE', fontweight="bold", size=10)
    ax_lst[0][1].set_xlabel('EPISODES', size=10)

    ax_lst[1][0].plot(range(len(avg_valid_losses)), avg_valid_losses)
    ax_lst[1][0].set_title('AVG. VALIDATION LOSSES', fontweight="bold", size=10)

    ax_lst[1][1].plot(range(len(valid_accuracy_list)), valid_accuracy_list)
    ax_lst[1][1].set_title('VALIDATION ACCURACY CHANGE', fontweight="bold", size=10)
    ax_lst[1][1].set_xlabel('EPISODES', size=10)

    filename = os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type, 'graphs', "{0}_{1}_{2:.2f}_{3:.2f}_{4}_{5:.2f}.png".format(
        coin_name,
        last_save_epoch,
        valid_loss_min,
        last_valid_accuracy,
        valid_size,
        one_count_rate
    ))

    plt.savefig(filename)
    plt.close('all')


def train(optimizer, model, criterion, train_losses, x_train_normalized, y_up_train):
    model.train()
    optimizer.zero_grad()
    out = model.forward(x_train_normalized)

    loss = criterion(out, y_up_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    out = torch.sigmoid(out)
    t = torch.tensor(0.5).to(DEVICE)
    output_index = (out > t).float() * 1

    return y_up_train.size(0), (output_index == y_up_train).sum().float()


def post_train_processing(train_losses, avg_train_losses, train_accuracy_list, correct, total_size):
    train_loss = np.average(train_losses)
    avg_train_losses.append(train_loss)

    train_accuracy = 100 * correct / total_size
    train_accuracy_list.append(train_accuracy)

    return train_loss, train_accuracy


def validate(epoch, model, criterion, valid_losses, x_valid_normalized, y_up_valid):
    model.eval()
    out = model.forward(x_valid_normalized)
    loss = criterion(out, y_up_valid)
    valid_losses.append(loss.item())

    out = torch.sigmoid(out)
    t = torch.tensor(0.5).to(DEVICE)
    output_index = (out > t).float() * 1

    if VERBOSE: logger.info("Epoch {0} - Y_pred: {1}, Y_true: {2}".format(epoch, output_index, y_up_valid))

    return y_up_valid.size(0), (output_index == y_up_valid).sum().float()


def post_validation_processing(valid_losses, avg_valid_losses, valid_accuracy_list, correct, total_size):
    valid_loss = np.average(valid_losses)
    avg_valid_losses.append(valid_loss)

    valid_accuracy = 100 * correct / total_size
    valid_accuracy_list.append(valid_accuracy)

    return valid_loss, valid_accuracy


def make_model(model_type, x_normalized, y_up, global_total_size, X_train, y_train, X_test, y_test, one_rate):
    if VERBOSE:
        logger.info("[[[{0}]]]".format(model_type))

    model = load_model(model_type=model_type)
    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        best_model = model
        best_f1_score = f1_score(y_test, y_pred)
    else:
        gc.collect()
        coin_model_start_time = time.time()

        X = x_normalized.reshape(global_total_size, -1)
        y = y_up

        if model_type == "GB":
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': np.linspace(1, 8, 4, endpoint=True),
                'n_estimators': [32, 64, 128],
                'max_features': list(range(int(x_normalized.shape[1] / 2), x_normalized.shape[1], 2)),
            }

            model_classifier = GradientBoostingClassifier()
        elif model_type == "XGBOOST":
            param_grid = {
                'objective': ['binary:logistic'],
                'max_depth': [4, 6, 8],
                'min_child_weight': [9, 11],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.7],
                'n_estimators': [50, 100, 200]
            }
            model_classifier = xgb.XGBClassifier()
        else:
            param_grid = {}
            model_classifier = None

        clf = GridSearchCV(
            model_classifier,
            param_grid,
            cv=StratifiedKFold(n_splits=4, shuffle=True),
            scoring='f1',
            refit=True,
            verbose=True
        )

        clf.fit(X, y)

        coin_model_elapsed_time = time.time() - coin_model_start_time
        coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))

        msg_str = "{0} - make_model - Elapsed Time: {1}, Best Score: {2}\n".format(
            model_type, coin_model_elapsed_time_str, clf.best_score_
        )
        logger.info(msg_str)
        logger.info(clf.best_params_)
        if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)

        best_model = clf.best_estimator_
        best_f1_score = clf.best_score_

    model_filename = save_model(best_model, model_type=model_type)

    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")

    model = trade_db_session.query(Model).filter(Model.model_type == model_type).first()

    if model:
        model.model_filename = model_filename
        model.one_rate = float(one_rate)
        model.train_size = model.train_size + global_total_size
        model.datetime = dt.datetime.strptime(current_time_str, fmt.replace("T", " "))
        model.window_size = WINDOW_SIZE
        model.future_target_size = FUTURE_TARGET_SIZE
        model.up_rate = UP_RATE
        model.feature_size = INPUT_SIZE
        model.best_score = float(best_f1_score)

        trade_db_session.commit()
    else:
        model = Model(
            model_type=model_type,
            model_filename=model_filename,
            one_rate=float(one_rate),
            train_size=global_total_size,
            datetime=dt.datetime.strptime(current_time_str, fmt.replace("T", " ")),
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE,
            feature_size=INPUT_SIZE,
            best_score=float(best_f1_score)
        )
        trade_db_session.add(model)
        trade_db_session.commit()

    if VERBOSE:
        msg_str = "TRAINED_SIZE: {0}, BEST_SCORE: {1}\n".format(model.train_size, model.best_score)
        logger.info(msg_str)
        SLACK.send_message("me", msg_str)


def main(coin_names):
    start_time = time.time()

    heading_msg = "\n**************************\n"
    heading_msg += "WINDOW SIZE:{0}, FUTURE_TARGET_SIZE:{1}, UP_RATE:{2}, INPUT_SIZE:{3}, DEVICE:{4}, MODEL_SOURCE:{5}".format(
        WINDOW_SIZE,
        FUTURE_TARGET_SIZE,
        UP_RATE,
        INPUT_SIZE,
        DEVICE,
        LOCAL_MODEL_SOURCE
    )
    logger.info(heading_msg)

    x_normalized_list = []
    y_up_list = []
    one_rate_list = []
    global_total_size = 0
    for idx, coin_name in enumerate(coin_names):
        gc.collect()

        upbit_order_book_data = UpbitOrderBookBasedData(coin_name)
        x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(split=False)

        if VERBOSE:
            logger.info("{0}, {1}: x_normalized_original: {2}, y_up_original: {3}, one_rate: {4}, total_size: {5}".format(
                idx,
                coin_name,
                x_normalized_original.shape,
                y_up_original.shape,
                one_rate,
                total_size
            ))

        x_normalized_list.append(x_normalized_original)
        y_up_list.append(y_up_original)
        one_rate_list.append(one_rate)
        global_total_size += total_size

    gc.collect()
    x_normalized = np.concatenate(x_normalized_list)

    gc.collect()
    y_up = np.concatenate(y_up_list)

    one_rate = float(np.mean(one_rate_list))

    if VERBOSE:
        logger.info("TOTAL -- x_normalized: {0}, y_up: {1}, one_rate: {2}, total_size: {3}".format(
            x_normalized.shape, y_up.shape, one_rate, global_total_size
        ))

    ##### MAKE MODELS
    X_train, X_test, y_train, y_test = train_test_split(x_normalized, y_up, test_size=0.2, random_state=0)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    ## GB
    make_model("GB", x_normalized, y_up, global_total_size, X_train, y_train, X_test, y_test, one_rate)

    ## XGBOOST
    make_model("XGBOOST", x_normalized, y_up, global_total_size, X_train, y_train, X_test, y_test, one_rate)

    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    logger.info("####################################################################")
    logger.info("Elapsed Time: {0}".format(elapsed_time_str))
    logger.info("####################################################################\n")

    slack_msg = "MODEL CONSTRUCTION COMES TO END: - ELAPSED_TIME:{0} @ {1}".format(
        elapsed_time_str, SOURCE
    )
    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", slack_msg)


if __name__ == "__main__":
    mkdir_models()

    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", "MAKE MODELS STARTED @ {0}".format(SOURCE))

    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    while True:
        main(coin_names=upbit.get_all_coin_names(parts=1))
