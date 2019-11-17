# https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn
import glob
import time

import sys, os
import xgboost as xgb
from pytz import timezone

from sklearn.metrics import classification_report, precision_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
import matplotlib.pyplot as plt
from common.utils import *

from predict.model_rnn import LSTM
import numpy as np
import os
from common.logger import get_logger
from upbit.upbit_api import Upbit
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
from common.utils import save_model

import torch.nn.modules.loss

logger = get_logger("make_models")

if os.getcwd().endswith("predict"):
    os.chdir("..")

engine_model = create_engine('sqlite:///{0}/web/db/model.db'.format(PROJECT_HOME))
Base = declarative_base()


class Model(Base):
    __tablename__ = 'model'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    coin_name = Column(String)
    model_type = Column(String)
    model_filename = Column(String)
    window_size = Column(Integer)
    future_target_size = Column(Integer)
    up_rate = Column(Float)
    feature_size = Column(Integer)
    datetime = Column(String)
    one_rate = Column(Float)
    train_size = Column(Integer)

    def __repr__(self):
        return "[ID: {0}] Coin Name: {1} Model Type: {2}, Model Filename: {3}, Datetime: {4}, One Rate: {5}, Train Size: {6}".format(
            self.id, self.coin_name, self.model_type, self.model_filename, self.datetime, self.one_rate, self.train_size
        )


Base.metadata.create_all(engine_model)
Session = sessionmaker(bind=engine_model)
session = Session()


def mkdir_models():
    if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE)):
        os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE))

    model_type_list = ['LSTM', 'GB', 'XGBOOST']

    for model_type in model_type_list:
        if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type)):
            os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type))

        if not os.path.exists(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type, 'graphs')):
            os.makedirs(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, model_type, 'graphs'))

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


def make_gboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate):
    coin_model_start_time = time.time()

    X = x_normalized_original.numpy().reshape(total_size, -1)
    y = y_up_original.numpy()

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': np.linspace(1, 8, 4, endpoint=True),
        'n_estimators': [32, 64, 128],
        'max_features': list(range(int(x_normalized_original.shape[1] / 2), x_normalized_original.shape[1], 2)),
    }

    gb_model = GradientBoostingClassifier()

    clf = GridSearchCV(
        gb_model,
        param_grid,
        cv=StratifiedKFold(n_splits=4, shuffle=True),
        scoring='accuracy',
        refit=True,
        verbose=True
    )

    clf.fit(X, y)

    logger.info(clf.best_estimator_)

    coin_model_elapsed_time = time.time() - coin_model_start_time
    coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))

    msg_str = "{0}: GradientBoostingClassifier - make_gboost_model - Elapsed Time: {1}\n".format(coin_name, coin_model_elapsed_time_str)
    logger.info(msg_str)
    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)

    return clf.best_estimator_


def make_lstm_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate):
    coin_model_start_time = time.time()

    lstm_model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    early_stopping = EarlyStopping(patience=15)

    param_grid = {
        'module': [lstm_model],
        'max_epochs': [500],
        'lr': [0.01, 0.05, 0.1],
        'module__bias': [True, False],
        'module__dropout': [0.0, 0.25, 0.5],
        'optimizer': [torch.optim.Adam],
        'device': [DEVICE],
        'callbacks': [[auc, early_stopping]]
    }

    X = x_normalized_original.numpy()
    y = y_up_original.numpy().astype(np.int64)

    lstm_model = LSTM(input_size=INPUT_SIZE, bias=True, dropout=0.5).to(DEVICE)
    net = NeuralNetClassifier(
        lstm_model,
        max_epochs=10,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        device=DEVICE
    )

    clf = GridSearchCV(
        net,
        param_grid,
        cv=StratifiedKFold(n_splits=4, shuffle=True),
        scoring='roc_auc',
        refit=True,
        verbose=True
    )

    clf.fit(X, y)

    logger.info(clf.best_estimator_)

    coin_model_elapsed_time = time.time() - coin_model_start_time
    coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))

    msg_str = "{0}: LSTMClassifier - make_lstm_model - Elapsed Time: {1}\n".format(coin_name, coin_model_elapsed_time_str)
    logger.info(msg_str)
    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)

    return clf.best_estimator_


def make_xgboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate):
    coin_model_start_time = time.time()

    X = x_normalized_original.numpy().reshape(total_size, -1)
    y = y_up_original.numpy()

    param_grid = {
        'objective': ['binary:logistic'],
        'max_depth': [4, 6, 8],
        'min_child_weight': [9, 11],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7],
        'n_estimators': [50, 100, 200]
    }
    xgb_model = xgb.XGBClassifier()

    clf = GridSearchCV(
        xgb_model,
        param_grid,
        cv=StratifiedKFold(n_splits=4, shuffle=True),
        scoring='accuracy',
        refit=True,
        verbose=True
    )

    clf.fit(X, y)

    logger.info(clf.best_estimator_)

    coin_model_elapsed_time = time.time() - coin_model_start_time
    coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))

    msg_str = "{0}: XGBoostClassifier - make_xgboost_model - Elapsed Time: {1}\n".format(coin_name, coin_model_elapsed_time_str)
    logger.info(msg_str)
    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)

    return clf.best_estimator_


def main(coin_names, model_source):
    start_time = time.time()

    heading_msg = "\n**************************\n"
    heading_msg += "WINDOW SIZE:{0}, FUTURE_TARGET_SIZE:{1}, UP_RATE:{2}, INPUT_SIZE:{3}, DEVICE:{4}, MODEL_SOURCE:{5}".format(
        WINDOW_SIZE,
        FUTURE_TARGET_SIZE,
        UP_RATE,
        INPUT_SIZE,
        DEVICE,
        model_source
    )
    logger.info(heading_msg)

    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")

    for i, coin_name in enumerate(coin_names):
        upbit_order_book_data = UpbitOrderBookBasedData(coin_name)

        x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(split=False)

        if VERBOSE:
            logger.info("{0}: x_normalized_original: {1}, y_up_original: {2}, one_rate: {3}, total_size: {4}".format(
                coin_name,
                x_normalized_original.size(),
                y_up_original.size(),
                one_rate,
                total_size
            ))

        if one_rate == -1 or total_size == -1:
            if VERBOSE:
                logger.info("{0}: 'One rate' is too low, so that the model construction is skipped".format(coin_name))
            continue
        else:
            # if VERBOSE:
            #     logger.info("[[[LSTM]]]")
            # best_model = make_lstm_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate)
            # save_model(coin_name, best_model, model_type="LSTM")

            if VERBOSE:
                logger.info("[[[XGBoost]]]")
            best_model = make_xgboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate)
            model_filename = save_model(coin_name, best_model, model_type="XGBOOST")

            model = session.query(Model).filter(Model.coin_name == coin_name and Model.model_type == 'XGBOOST')

            logger.info("{0}: {1}".format(coin_name, model))

            if model:
                model.model_filename=model_filename
                model.one_rate=one_rate
                model.train_size=total_size
                model.datetime=current_time_str
                model.window_size=WINDOW_SIZE
                model.future_target_size=FUTURE_TARGET_SIZE
                model.up_rate=UP_RATE
                model.feature_size=INPUT_SIZE

                session.commit()
            else:
                model = Model(
                    coin_name=coin_name,
                    model_type="XGBOOST",
                    model_filename=model_filename,
                    one_rate=one_rate,
                    train_size=total_size,
                    datetime=current_time_str,
                    window_size=WINDOW_SIZE,
                    future_target_size=FUTURE_TARGET_SIZE,
                    up_rate=UP_RATE,
                    feature_size=INPUT_SIZE
                )
                session.add(model)
                session.commit()

            if VERBOSE:
                logger.info("[[[Gradient Boosting]]]")
            best_model = make_gboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate)
            model_filename = save_model(coin_name, best_model, model_type="GB")

            model = session.query(Model).filter(Model.coin_name == coin_name and Model.model_type == 'GB')

            logger.info("{0}: {1}".format(coin_name, model))
            
            if model:
                model.model_filename = model_filename
                model.one_rate = one_rate
                model.train_size = total_size
                model.datetime = current_time_str
                model.window_size = WINDOW_SIZE
                model.future_target_size = FUTURE_TARGET_SIZE
                model.up_rate = UP_RATE
                model.feature_size = INPUT_SIZE

                session.commit()
            else:
                model = Model(
                    coin_name=coin_name,
                    model_type="GB",
                    model_filename=model_filename,
                    one_rate=one_rate,
                    train_size=total_size,
                    datetime=current_time_str,
                    window_size=WINDOW_SIZE,
                    future_target_size=FUTURE_TARGET_SIZE,
                    up_rate=UP_RATE,
                    feature_size=INPUT_SIZE
                )
                session.add(model)
                session.commit()

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
        main(coin_names=upbit.get_all_coin_names(), model_source=LOCAL_MODEL_SOURCE)


# def get_best_model_by_nested_cv(coin_name, X, y, inner_cv, outer_cv, Classifier, parameter_grid):
#     outer_score_list = []
#     best_param_list = []
#     model_list = []
#
#     num_outer_split = 1
#     for training_samples_idx, test_samples_idx in outer_cv.split(X, y):
#         logger.info("COIN_NAME: {0} - [Outer Split: #{0}]".format(coin_name, num_outer_split))
#         best_score = -np.inf
#
#         for parameters in parameter_grid:
#             # print("Parameters: {0}".format(parameters))
#             cv_scores = []
#             num_inner_split = 1
#             for inner_train_idx, inner_test_idx in inner_cv.split(X[training_samples_idx], y[training_samples_idx]):
#                 clf = Classifier(**parameters)
#                 print(X[inner_train_idx])
#                 print(y[inner_train_idx])
#
#                 clf.fit(X[inner_train_idx], y[inner_train_idx])
#                 score = clf.score(X[inner_test_idx], y[inner_test_idx])
#
#                 y_true, y_pred = y[inner_test_idx], clf.predict(X[inner_test_idx])
#                 print(y_true)
#                 print(y_pred)
#                 print(precision_score(y_true, y_pred))
#
#                 cv_scores.append(score)
#                 #                 print("Inner Split: #{0}, Score: #{1}".format(
#                 #                     num_inner_split,
#                 #                     score
#                 #                 ))
#                 num_inner_split += 1
#
#             mean_score = np.mean(cv_scores)
#             if mean_score > best_score:
#                 best_score = mean_score
#                 best_params = parameters
#                 # print("Mean Score:{0}, Best Score:{1}".format(mean_score, best_score))
#
#         logger.info("COIN_NAME: {0} - Outer Split: #{1}, Best Score: {2}, Best Parameter: #{3}".format(
#             coin_name,
#             num_outer_split,
#             best_score,
#             best_params
#         ))
#
#         clf = Classifier(**best_params)
#         clf.fit(X[training_samples_idx], y[training_samples_idx])
#
#         best_param_list.append(best_params)
#         outer_score_list.append(clf.score(X[test_samples_idx], y[test_samples_idx]))
#         model_list.append(clf)
#
#         num_outer_split += 1
#
#     best_score = -np.inf
#     best_model = None
#     for idx, score in enumerate(outer_score_list):
#         if score > best_score:
#             best_score = score
#             best_model = model_list[idx]
#
#     return best_model
