# https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
import glob
import time
import torch.nn as nn

from common.global_variables import *
import matplotlib.pyplot as plt

from predict.model_rnn import LSTM
from predict.model_cnn import CNN
from predict.early_stopping import EarlyStopping
import numpy as np
import os
from common.logger import get_logger
from upbit.upbit_orderbook_based_data import UpbitOrderBookBasedData, get_data_loader

logger = get_logger("make_models_logger")

if os.getcwd().endswith("predict"):
    os.chdir("..")


def mkdir_models():
    if not os.path.exists("./models/"):
        os.makedirs("./models/")

    if not os.path.exists("./models/CNN"):
        os.makedirs("./models/CNN")

    if not os.path.exists("./models/CNN/graphs"):
        os.makedirs("./models/CNN/graphs")

    if not os.path.exists("./models/LSTM"):
        os.makedirs("./models/LSTM")

    if not os.path.exists("./models/LSTM/graphs"):
        os.makedirs("./models/LSTM/graphs")

    # files = glob.glob('./{0}/*'.format(filename))
    # for f in files:
    #     if os.path.isfile(f):
    #         os.remove(f)


def save_graph(coin_name, model_type, valid_loss_min, last_valid_accuracy, last_save_epoch, valid_size, one_count_rate, avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list):
    files = glob.glob('./models/{0}/graphs/{1}*'.format(model_type, coin_name))
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

    filename = "./models/{0}/graphs/{1}_{2}_{3:.2f}_{4:.2f}_{5}_{6:.2f}.png".format(
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
    t = torch.Tensor([0.5]).to(DEVICE)
    output_index = (out > t).float()
    output_index = output_index * 1

    return y_up_train.size(0), (output_index == y_up_train).sum().float()


def post_train_processing(train_losses, avg_train_losses, train_accuracy_list, correct, total):
    train_loss = np.average(train_losses)
    avg_train_losses.append(train_loss)

    train_accuracy = 100 * correct / total
    train_accuracy_list.append(train_accuracy)

    return train_loss, train_accuracy


def validate(epoch, model, criterion, valid_losses, x_valid_normalized, y_up_valid):
    model.eval()
    out = model.forward(x_valid_normalized)
    loss = criterion(out, y_up_valid)
    valid_losses.append(loss.item())

    out = torch.sigmoid(out)
    t = torch.Tensor([0.5]).to(DEVICE)
    output_index = (out > t).float()
    output_index = output_index * 1

    if VERBOSE: logger.info("{0}: Predict - {1}, Y - {2}".format(epoch, output_index, y_up_valid))

    return y_up_valid.size(0), (output_index == y_up_valid).sum().float()


def post_validation_processing(valid_losses, avg_valid_losses, valid_accuracy_list, correct, total):
    valid_accuracy = 100 * correct / total
    valid_accuracy_list.append(valid_accuracy)

    valid_loss = np.average(valid_losses)
    avg_valid_losses.append(valid_loss)

    return valid_loss, valid_accuracy


def main(model_type):
    start_time = time.time()

    coin_names_high_quality_models = []

    batch_size = 6
    lr = 0.001

    heading_msg = "**************************\n"
    heading_msg += "{0} Model - WINDOW SIZE:{1}, FUTURE_TARGET_SIZE:{2}, UP_RATE:{3}, INPUT_SIZE:{4}, DEVICE:{5}".format(
        model_type,
        WINDOW_SIZE,
        FUTURE_TARGET_SIZE,
        UP_RATE,
        INPUT_SIZE,
        DEVICE
    )

    logger.info(heading_msg)

    patience = 50

    coin_names = UPBIT.get_all_coin_names()

    for i, coin_name in enumerate(coin_names):
        upbit_data = UpbitOrderBookBasedData(coin_name)

        x_train_original, x_train_normalized_original, y_train_original, y_train_normalized_original, y_up_train_original, \
        one_rate_train, train_size, \
        x_valid_original, x_valid_normalized_original, y_valid_original, y_valid_normalized_original, y_up_valid_original, \
        one_rate_valid, valid_size = upbit_data.get_data(model_type=model_type)

        if VERBOSE:
            msg = "{0:>2} [{1:>5}] Train Size:{2:>3d}/{3:>3}[{4:.4f}], Validation Size:{5:>3d}/{6:>3}[{7:.4f}]".format(
                i,
                coin_name,
                int(y_up_train_original.sum()),
                train_size,
                one_rate_train,
                int(y_up_valid_original.sum()),
                valid_size,
                one_rate_valid
            )
            logger.info(msg)

        if model_type == "CNN":
            model = CNN(input_width=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
        elif model_type == "LSTM":
            model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
        else:
            pass

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses = []
        valid_losses = []

        avg_train_losses = []
        avg_valid_losses = []

        train_accuracy_list = []
        valid_accuracy_list = []

        early_stopping = EarlyStopping(model_type=model_type, coin_name=coin_name, patience=patience, verbose=VERBOSE, logger=logger)

        early_stopped = False
        for epoch in range(1, NUM_EPOCHS + 1):
            x_train = x_train_original.clone().detach()
            x_train_normalized = x_train_normalized_original.clone().detach()
            y_train = y_train_original.clone().detach()
            y_train_normalized = y_train_normalized_original.clone().detach()
            y_up_train = y_up_train_original.clone().detach()

            train_data_loader = get_data_loader(
                x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, batch_size=batch_size, suffle=True
            )

            correct = 0.0
            total = 0.0

            # training
            for x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, num_batches in train_data_loader:
                total_batch, correct_batch = train(
                    optimizer, model, criterion, train_losses,
                    x_train_normalized, y_up_train
                )
                total += total_batch
                correct += correct_batch

            train_loss, train_accuracy = post_train_processing(
                train_losses, avg_train_losses, train_accuracy_list, correct, total
            )

            # validation
            # 배치정규화나 드롭아웃은 학습할때와 테스트 할때 다르게 동작하기 때문에 모델을 evaluation 모드로 바꿔서 테스트함.
            x_valid = x_valid_original.clone().detach()
            x_valid_normalized = x_valid_normalized_original.clone().detach()
            y_valid = y_valid_original.clone().detach()
            y_valid_normalized = y_valid_normalized_original.clone().detach()
            y_up_valid = y_up_valid_original.clone().detach()

            valid_data_loader = get_data_loader(
                x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, batch_size=batch_size, suffle=False
            )

            correct = 0.0
            total = 0.0

            for x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, num_batches in valid_data_loader:
                total_batch, correct_batch = validate(
                    epoch, model, criterion, valid_losses, x_valid_normalized, y_up_valid
                )
                total += total_batch
                correct += correct_batch

            if VERBOSE: logger.info("\n")

            valid_loss, valid_accuracy = post_validation_processing(
                valid_losses, avg_valid_losses, valid_accuracy_list, correct, total
            )

            print_msg = "{0} - Epoch[{1}/{2}] - \n  t_loss:{3:.6f},   t_accuracy:{4:.2f},   v_loss:{5:.6f},   v_accuracy:{6:.2f}".format(
                coin_name,
                epoch,
                NUM_EPOCHS,
                train_loss,
                train_accuracy,
                valid_loss,
                valid_accuracy
            )

            if VERBOSE: logger.info(print_msg)

            early_stopping(valid_loss, valid_accuracy, epoch, model, valid_size, one_rate_valid)

            if early_stopping.early_stop:
                early_stopped = True
                if VERBOSE: logger.info("Early stopping @ Epoch {0}: Last Save Epoch {1}".format(epoch, early_stopping.last_save_epoch))
                break

        if (not early_stopped) and VERBOSE:
            logger.info("Normal Stopping @ Epoch {0}: Last Save Epoch {1}".format(NUM_EPOCHS, early_stopping.last_save_epoch))

        high_quality_model_condition_list = [
            early_stopping.min_valid_loss < MIN_VALID_LOSS_THRESHOLD,
            early_stopping.last_valid_accuracy > LAST_VALID_ACCURACY_THRESHOLD,
            early_stopping.last_save_epoch > LAST_SAVE_EPOCH_THRESHOLD,
            one_rate_valid > ONE_RATE_VALID_THRESHOLD
        ]

        msg = "Last Save Epoch: {0:3d} - Min of Valid Loss: {1:.4f}, Last Valid Accuracy:{2:.4f} - {3}".format(
            early_stopping.last_save_epoch,
            early_stopping.min_valid_loss,
            early_stopping.last_valid_accuracy,
            all(high_quality_model_condition_list)
        )
        logger.info(msg)

        if all(high_quality_model_condition_list):
            coin_names_high_quality_models.append(coin_name)

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

    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    logger.info("####################################################################")
    logger.info("Coin Name with High Quality Model: {0}".format(coin_names_high_quality_models))
    logger.info("Elapsed Time: {0}".format(elapsed_time_str))
    logger.info("####################################################################")

    slack_msg = "MODEL TYPE:{0} - HIGH QUALITY MODELS:{1} - ELAPSED_TIME:{2} @ {3}".format(
        model_type, coin_names_high_quality_models, elapsed_time_str, SOURCE
    )
    SLACK.send_message("me", slack_msg)


if __name__ == "__main__":
    mkdir_models()

    SLACK.send_message("me", "MAKE MODELS STARTED @ {0}".format(SOURCE))

    main(model_type="CNN")
    main(model_type="LSTM")

