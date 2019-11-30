import glob
import os
import subprocess
import sys

from codes.upbit.upbit_api import Upbit

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import REMOTE_SOURCE_HOST, REMOTE_SOURCE, CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt, \
    LOCAL_MODEL_SOURCE
from common.global_variables import SSH_SCP_SOURCE_PORT, SSH_SCP_SOURCE_ID, SSH_SCP_SOURCE_PASSWORD, LOCAL_TARGET
from common.global_variables import PROJECT_HOME
from common.logger import get_logger

logger = get_logger("pull_models")


def check_remote_file(coin_name):
    output = subprocess.getoutput("sshpass -p{0} ssh -p {1} -o StrictHostKeyChecking=no {2}@{3} 'ls {4}{5}/{6}_*'".format(
        SSH_SCP_SOURCE_PASSWORD,
        SSH_SCP_SOURCE_PORT,
        SSH_SCP_SOURCE_ID,
        REMOTE_SOURCE_HOST,
        REMOTE_SOURCE,
        "LSTM",
        coin_name
    ))
    return output


def download_remote_file(remote_file, local_file):
    subprocess.getoutput("sshpass -p{0} scp -P {1} -o StrictHostKeyChecking=no {2}@{3}:{4} {5}{6}/{7}".format(
        SSH_SCP_SOURCE_PASSWORD,
        SSH_SCP_SOURCE_PORT,
        SSH_SCP_SOURCE_ID,
        REMOTE_SOURCE_HOST,
        remote_file,
        LOCAL_TARGET,
        "LSTM",
        local_file
    ))


if __name__ == "__main__":
    logger.info("\n#######################################################################\n")
    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    coin_names = upbit.get_all_coin_names()
    lstm_model_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'LSTM', '*.pt'))

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
