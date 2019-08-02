import glob
import os
import subprocess
import sys

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import REMOTE_SOURCE_HOST, REMOTE_SOURCE
from common.global_variables import SSH_SCP_SOURCE_PORT, SSH_SCP_SOURCE_ID, SSH_SCP_SOURCE_PASSWORD, LOCAL_TARGET
from common.global_variables import UPBIT, PROJECT_HOME
from common.logger import get_logger

logger = get_logger("pull_models_logger")


def check_remote_file(model_type, coin_name):
    output = subprocess.getoutput("sshpass -p{0} ssh -p {1} -o StrictHostKeyChecking=no {2}@{3} 'ls {4}{5}/{6}_*'".format(
        SSH_SCP_SOURCE_PASSWORD,
        SSH_SCP_SOURCE_PORT,
        SSH_SCP_SOURCE_ID,
        REMOTE_SOURCE_HOST,
        REMOTE_SOURCE,
        model_type,
        coin_name
    ))
    return output


def download_remote_file(model_type, remote_file, local_file):
    subprocess.getoutput("sshpass -p{0} scp -P {1} -o StrictHostKeyChecking=no {2}@{3}:{4} {5}{6}/{7}".format(
        SSH_SCP_SOURCE_PASSWORD,
        SSH_SCP_SOURCE_PORT,
        SSH_SCP_SOURCE_ID,
        REMOTE_SOURCE_HOST,
        remote_file,
        LOCAL_TARGET,
        model_type,
        local_file
    ))


if __name__ == "__main__":
    logger.info("\n#######################################################################\n")
    coin_names = UPBIT.get_all_coin_names()
    cnn_model_files = glob.glob(PROJECT_HOME + 'models/CNN/*.pt')
    lstm_model_files = glob.glob(PROJECT_HOME + 'models/LSTM/*.pt')

    for coin_name in coin_names:
        remote_file = check_remote_file("CNN", coin_name)
        if ".pt" in remote_file:
            for cnn_model_file in cnn_model_files:
                if "CNN/{0}_".format(coin_name) in cnn_model_file:
                    os.remove(cnn_model_file)
                    logger.info("{0} is removed.".format(cnn_model_file))
                    break
            download_remote_file("CNN", remote_file, remote_file.split("/")[-1])
            logger.info("CNN {0} is downloaded.".format(remote_file))

    for coin_name in coin_names:
        remote_file = check_remote_file("LSTM", coin_name)
        if ".pt" in remote_file:
            for lstm_model_file in lstm_model_files:
                if "LSTM/{0}_".format(coin_name) in lstm_model_file:
                    os.remove(lstm_model_file)
                    logger.info("{0} is removed.".format(lstm_model_file))
                    break
            download_remote_file("LSTM", remote_file, remote_file.split("/")[-1])
            logger.info("LSTM {0} is downloaded.".format(remote_file))
