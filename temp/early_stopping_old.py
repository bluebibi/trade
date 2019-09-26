#https://github.com/Bjarten/early-stopping-pytorch
import glob
import os
import subprocess

import numpy as np
import torch

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import SSH_SCP_TARGET_PEM_FILE_PATH, SSH_SCP_TARGET_ID, REMOTE_TARGET_HOST, REMOTE_TARGET, \
    IS_PUSH_AFTER_MAKE_MODELS, SELF_MODELS_MODE, SELF_MODEL_SOURCE, LOCAL_MODEL_SOURCE

if SELF_MODELS_MODE:
    model_source = SELF_MODEL_SOURCE
else:
    model_source = LOCAL_MODEL_SOURCE


class EarlyStoppingOld:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, coin_name, patience=7, verbose=False, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.min_valid_loss = np.Inf
        self.coin_name = coin_name
        self.last_save_epoch = -1
        self.last_filename = None
        self.last_valid_accuracy = -1
        self.last_state_dict = None
        self.logger = logger

    def __call__(self, valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate):
        if epoch > 1:
            if self.min_valid_loss is np.Inf:
                self.save_checkpoint(valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate)
            elif valid_loss >= self.min_valid_loss:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} @ Epoch {epoch}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate)
                self.counter = 0
        else:
            pass

    def save_checkpoint(self, valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'{self.coin_name}: Saving Model @ Epoch {epoch} - v_loss decreased '
                             f'({self.min_valid_loss:.4f} --> {valid_loss:.4f}) - v_accuracy {valid_accuracy:.6f}\n')

        new_filename = "{0}_{1}_{2:.2f}_{3:.2f}_{4}_{5:.2f}.pt".format(
            self.coin_name,
            epoch,
            valid_loss,
            valid_accuracy,
            valid_size,
            one_count_rate
        )

        self.last_state_dict = model.state_dict()

        self.last_filename = new_filename
        self.min_valid_loss = valid_loss
        self.last_save_epoch = epoch
        self.last_valid_accuracy = valid_accuracy

    def save_last_model(self):
        files = glob.glob(PROJECT_HOME + '{0}LSTM/{1}*'.format(model_source, self.coin_name))
        for f in files:
            os.remove(f)

        file_name = "{0}LSTM/{1}".format(model_source, self.last_filename)
        torch.save(self.last_state_dict, file_name)

        if IS_PUSH_AFTER_MAKE_MODELS:
            self.push_models(file_name)

    def push_models(self, file_name):
        cmd = "ssh -i {0} {1}@{2} 'ls {3}LSTM/{4}_*'".format(
            SSH_SCP_TARGET_PEM_FILE_PATH,
            SSH_SCP_TARGET_ID,
            REMOTE_TARGET_HOST,
            REMOTE_TARGET,
            self.coin_name
        )

        self.logger.info(cmd)

        remote_file = subprocess.getoutput(cmd)

        if ".pt" in remote_file:
            cmd = "ssh -i {0} {1}@{2} 'rm {3}'".format(
                SSH_SCP_TARGET_PEM_FILE_PATH,
                SSH_SCP_TARGET_ID,
                REMOTE_TARGET_HOST,
                remote_file
            )
            self.logger.info(cmd)
            subprocess.getoutput(cmd)

        cmd = "scp -i {0} {1} {2}@{3}:{4}LSTM/".format(
            SSH_SCP_TARGET_PEM_FILE_PATH,
            file_name,
            SSH_SCP_TARGET_ID,
            REMOTE_TARGET_HOST,
            REMOTE_TARGET
        )
        self.logger.info(cmd)
        subprocess.getoutput(cmd)

