# https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
import glob
import time
import torch.nn as nn

import sys, os

from skorch import NeuralNetClassifier

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
import matplotlib.pyplot as plt

from predict.model_rnn import LSTM
from predict.early_stopping import EarlyStopping
import numpy as np
import os
from common.logger import get_logger
from upbit.upbit_api import Upbit
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData, get_data_loader
from common.utils import save_gb_model

logger = get_logger("make_models")

if os.getcwd().endswith("predict"):
    os.chdir("..")

if SELF_MODELS_MODE:
    model_source = SELF_MODEL_SOURCE
else:
    model_source = LOCAL_MODEL_SOURCE


def mkdir_models(source):
    if not os.path.exists(PROJECT_HOME + "{0}".format(source)):
        os.makedirs(PROJECT_HOME + "{0}".format(source))

    if not os.path.exists(PROJECT_HOME + "{0}LSTM".format(source)):
        os.makedirs(PROJECT_HOME + "{0}LSTM".format(source))

    if not os.path.exists(PROJECT_HOME + "{0}LSTM/graphs".format(source)):
        os.makedirs(PROJECT_HOME + "{0}LSTM/graphs".format(source))

    if not os.path.exists(PROJECT_HOME + "{0}GB".format(source)):
        os.makedirs(PROJECT_HOME + "{0}GB".format(source))

    if not os.path.exists(PROJECT_HOME + "{0}GB/graphs".format(source)):
        os.makedirs(PROJECT_HOME + "{0}GB/graphs".format(source))


def save_graph(coin_name, model_type, valid_loss_min, last_valid_accuracy, last_save_epoch, valid_size, one_count_rate,
               avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list):
    files = glob.glob(PROJECT_HOME + '{0}{1}/graphs/{2}*'.format(model_source, model_type, coin_name))
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

    filename = PROJECT_HOME + "{0}{1}/graphs/{2}_{3}_{4:.2f}_{5:.2f}_{6}_{7:.2f}.png".format(
        model_source,
        model_type,
        coin_name,
        last_save_epoch,
        valid_loss_min,
        last_valid_accuracy,
        valid_size,
        one_count_rate
    )

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


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier


def get_best_model_by_nested_cv(coin_name, X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_score_list = []
    best_param_list = []
    model_list = []

    num_outer_split = 1
    for training_samples_idx, test_samples_idx in outer_cv.split(X, y):
        logger.info("COIN_NAME: {0} - [Outer Split: #{0}]".format(coin_name, num_outer_split))
        best_score = -np.inf

        for parameters in parameter_grid:
            # print("Parameters: {0}".format(parameters))
            cv_scores = []
            num_inner_split = 1
            for inner_train_idx, inner_test_idx in inner_cv.split(X[training_samples_idx], y[training_samples_idx]):
                clf = Classifier(**parameters)
                clf.fit(X[inner_train_idx], y[inner_train_idx])
                score = clf.score(X[inner_test_idx], y[inner_test_idx])

                cv_scores.append(score)
                #                 print("Inner Split: #{0}, Score: #{1}".format(
                #                     num_inner_split,
                #                     score
                #                 ))
                num_inner_split += 1

            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters
                # print("Mean Score:{0}, Best Score:{1}".format(mean_score, best_score))

        logger.info("COIN_NAME: {0} - Outer Split: #{1}, Best Score: {2}, Best Parameter: #{3}".format(
            coin_name,
            num_outer_split,
            best_score,
            best_params
        ))

        clf = Classifier(**best_params)
        clf.fit(X[training_samples_idx], y[training_samples_idx])

        best_param_list.append(best_params)
        outer_score_list.append(clf.score(X[test_samples_idx], y[test_samples_idx]))
        model_list.append(clf)

        num_outer_split += 1

    best_score = -np.inf
    best_model = None
    for idx, score in enumerate(outer_score_list):
        if score > best_score:
            best_score = score
            best_model = model_list[idx]

    return best_score, best_model


def make_gboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate):
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_depth': np.linspace(1, 8, 4, endpoint=True),
    #     'n_estimators': [32, 64, 100, 200],
    #     'max_features': list(range(int(x_normalized_original.shape[1] / 2), x_normalized_original.shape[1], 2)),
    #     'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
    #     'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True)
    # }
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': np.linspace(1, 8, 4, endpoint=True),
        'n_estimators': [32, 64, 128],
        'max_features': list(range(int(x_normalized_original.shape[1] / 2), x_normalized_original.shape[1], 2)),
    }

    coin_model_start_time = time.time()

    X = x_normalized_original.numpy().reshape(total_size, -1)
    y = y_up_original.numpy()

    #     print("X.shape: {0}".format(X.shape))
    #     print("y.shape: {0}".format(y.shape))

    best_model = get_best_model_by_nested_cv(
        coin_name=coin_name,
        X=X,
        y=y,
        inner_cv=StratifiedKFold(n_splits=4, shuffle=True),
        outer_cv=StratifiedKFold(n_splits=4, shuffle=True),
        Classifier=GradientBoostingClassifier,
        parameter_grid=ParameterGrid(param_grid)
    )

    coin_model_elapsed_time = time.time() - coin_model_start_time
    coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))
    logger.info("==> {0}: GradientBoostingClassifier - make_gboost_model - Elapsed Time: {1}\n".format(coin_name, coin_model_elapsed_time_str))
    return best_model

def make_lstm_model(
        model, model_type, coin_name,
        x_train_normalized_original, y_up_train_original,
        x_valid_normalized_original, y_up_valid_original,
        valid_size, one_rate_valid):
    lstm_model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
    net = NeuralNetClassifier(
        lstm_model,
        max_epochs=10,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

def make_model(
        model, model_type, coin_name,
        x_train_normalized_original, y_up_train_original,
        x_valid_normalized_original, y_up_valid_original,
        valid_size, one_rate_valid):

    is_high_quality = False

    batch_size = 16
    lr = 0.001
    patience = 30

    coin_model_start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []

    train_accuracy_list = []
    valid_accuracy_list = []

    early_stopping = EarlyStopping(
        coin_name=coin_name, patience=patience, verbose=VERBOSE, logger=logger
    )

    early_stopped = False
    for epoch in range(1, NUM_EPOCHS + 1):
        x_train_normalized = x_train_normalized_original.clone().detach()
        y_up_train = y_up_train_original.clone().detach()

        train_data_loader = get_data_loader(
            x_train_normalized, y_up_train, batch_size=batch_size, shuffle=True
        )

        correct = 0.0
        total = 0.0

        # training
        for x_train_normalized, y_up_train, num_batches in train_data_loader:
            total_batch, correct_batch = train(
                optimizer, model, criterion, train_losses, x_train_normalized, y_up_train
            )
            total += total_batch
            correct += correct_batch

        train_loss, train_accuracy = post_train_processing(
            train_losses, avg_train_losses, train_accuracy_list, correct, total
        )

        # validation
        # 배치정규화나 드롭아웃은 학습할때와 테스트 할때 다르게 동작하기 때문에 모델을 evaluation 모드로 바꿔서 테스트함.
        x_valid_normalized = x_valid_normalized_original.clone().detach()
        y_up_valid = y_up_valid_original.clone().detach()

        valid_data_loader = get_data_loader(
            x_valid_normalized, y_up_valid, batch_size=batch_size, shuffle=False
        )

        correct = 0.0
        total = 0.0

        for x_valid_normalized, y_up_valid, num_batches in valid_data_loader:
            total_batch, correct_batch = validate(
                epoch, model, criterion, valid_losses, x_valid_normalized, y_up_valid
            )
            total += total_batch
            correct += correct_batch

        valid_loss, valid_accuracy = post_validation_processing(
            valid_losses, avg_valid_losses, valid_accuracy_list, correct, total
        )

        print_msg = "{0}-{1}:Epoch[{2}/{3}] - t_loss:{4:.4f}, t_accuracy:{5:.2f}, v_loss:{6:.4f}, " \
                    "v_accuracy:{7:.2f}".format(
            model_type, coin_name, epoch, NUM_EPOCHS, train_loss, train_accuracy, valid_loss, valid_accuracy
        )

        if VERBOSE: logger.info(print_msg)

        early_stopping(valid_loss, valid_accuracy, epoch, model, valid_size, one_rate_valid)

        if early_stopping.early_stop:
            early_stopped = True
            if VERBOSE: logger.info("Early stopping @ Epoch {0}: Last Save Epoch {1}".format(
                epoch, early_stopping.last_save_epoch
            ))
            break

    if (not early_stopped) and VERBOSE:
        logger.info("Normal Stopping @ Epoch {0}: Last Save Epoch {1}".format(
            NUM_EPOCHS, early_stopping.last_save_epoch
        ))

    high_quality_model_condition_list = [
        early_stopping.min_valid_loss < MIN_VALID_LOSS_THRESHOLD,
        early_stopping.last_valid_accuracy > LAST_VALID_ACCURACY_THRESHOLD,
        early_stopping.last_save_epoch > LAST_SAVE_EPOCH_THRESHOLD,
        one_rate_valid > ONE_RATE_VALID_THRESHOLD
    ]

    e_msg = "Last Save Epoch: {0:3d} - Min of Valid Loss: {1:.4f}, Last Valid Accuracy:{2:.4f} - {3}".format(
        early_stopping.last_save_epoch,
        early_stopping.min_valid_loss,
        early_stopping.last_valid_accuracy,
        all(high_quality_model_condition_list)
    )
    logger.info(e_msg)

    if all(high_quality_model_condition_list):
        is_high_quality = True

        save_graph(
            coin_name,
            model_type,
            early_stopping.min_valid_loss,
            early_stopping.last_valid_accuracy,
            early_stopping.last_save_epoch,
            valid_size, one_rate_valid,
            avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list
        )

        early_stopping.save_last_model()

        if VERBOSE: logger.info("\n")
    coin_model_elapsed_time = time.time() - coin_model_start_time
    coin_model_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(coin_model_elapsed_time))
    logger.info("==> {0}:{1} Elapsed Time: {2}\n".format(coin_name, model, coin_model_elapsed_time_str))

    return is_high_quality


def main(coin_names, model_source):
    start_time = time.time()

    heading_msg = "\n**************************\n"
    heading_msg += "WINDOW SIZE:{0}, FUTURE_TARGET_SIZE:{1}, UP_RATE:{2}, INPUT_SIZE:{3}, DEVICE:{4}, SELF_MODEL_MODE:{5}, MODEL_SOURCE:{6}".format(
        WINDOW_SIZE,
        FUTURE_TARGET_SIZE,
        UP_RATE,
        INPUT_SIZE,
        DEVICE,
        SELF_MODELS_MODE,
        model_source
    )
    logger.info(heading_msg)

    high_quality_models_lstm = []

    for i, coin_name in enumerate(coin_names):
        upbit_order_book_data = UpbitOrderBookBasedData(coin_name)

        # GradientBoosting
        x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(split=False)
        if VERBOSE:
            logger.info("[[[GradientBoosting]]]")
            logger.info("x_normalized_original: {0}, y_up_original: {1}, one_rate: {2}, total_size: {3}".format(
                x_normalized_original.size(),
                y_up_original.size(),
                one_rate,
                total_size
            ))
            best_model = make_gboost_model(coin_name, x_normalized_original, y_up_original, total_size, one_rate)
            save_gb_model(coin_name, best_model)


        # LSTM
        x_train_normalized_original, y_up_train_original, one_rate_train, train_size, \
        x_valid_normalized_original, y_up_valid_original, one_rate_valid, valid_size = upbit_order_book_data.get_dataset(split=True)

        if VERBOSE:
            logger.info("[[[LSTM]]]")
            t_msg = "{0:>2}-[{1:>5}] Train Size:{2:>3d}/{3:>3}[{4:.4f}], Valid Size:{5:>3d}/{6:>3}[{7:.4f}]".format(
                i,
                coin_name,
                int(y_up_train_original.sum()),
                train_size,
                one_rate_train,
                int(y_up_valid_original.sum()),
                valid_size,
                one_rate_valid
            )
            logger.info(t_msg)

        if one_rate_valid > ONE_RATE_VALID_THRESHOLD and valid_size > VALID_SIZE_THRESHOLD:
            model = LSTM(input_size=INPUT_SIZE).to(DEVICE)

            is_high_quality_lstm = make_model(
                model, "LSTM", coin_name,
                x_train_normalized_original, y_up_train_original, x_valid_normalized_original, y_up_valid_original,
                valid_size, one_rate_valid
            )

            if is_high_quality_lstm:
                high_quality_models_lstm.append(coin_name)
        else:
            logger.info("--> {0}: Model construction cancelled since 'one_rate_valid:{1}' or 'valid_size:{2}' is too low.\n".format(
                coin_name,
                one_rate_valid,
                valid_size
            ))

    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    logger.info("####################################################################")
    logger.info("Coin Name with High Quality LSTM Model: {0}".format(high_quality_models_lstm))
    logger.info("Elapsed Time: {0}".format(elapsed_time_str))
    logger.info("####################################################################\n")

    slack_msg = "HIGH QUALITY LSTM MODELS:{0} - ELAPSED_TIME:{1} @ {2}".format(
        high_quality_models_lstm, elapsed_time_str, SOURCE
    )
    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", slack_msg)


if __name__ == "__main__":
    mkdir_models(LOCAL_MODEL_SOURCE)
    mkdir_models(SELF_MODEL_SOURCE)

    if PUSH_SLACK_MESSAGE: SLACK.send_message("me", "MAKE MODELS STARTED @ {0}".format(SOURCE))

    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    while True:
        if SELF_MODELS_MODE:
            main(coin_names=upbit.get_all_coin_names(), model_source=SELF_MODEL_SOURCE)
        else:
            main(coin_names=upbit.get_all_coin_names(), model_source=LOCAL_MODEL_SOURCE)

    #main(coin_names=["OMG"])
