import glob
import logging, os, sys
from logging.handlers import RotatingFileHandler
from common.global_variables import PROJECT_HOME


def get_logger(name):
    """
    Args:
        name(str):생성할 log 파일명입니다.

    Returns:
         생성된 logger객체를 반환합니다.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if not os.path.exists(PROJECT_HOME + "/logs/"):
        os.makedirs(PROJECT_HOME + "/logs/")

    # files = glob.glob(PROJECT_HOME + "/logs/*")
    # for f in files:
    #     if os.path.isfile(f):
    #         os.remove(f)

    rotate_handler = RotatingFileHandler(
        PROJECT_HOME + "/logs/" + name + ".log",
        'a',
        1024 * 1024 * 5,
        5
    )
    formatter = logging.Formatter(
        '[%(levelname)s]-%(asctime)s-%(filename)s:%(lineno)s:%(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    rotate_handler.setFormatter(formatter)
    logger.addHandler(rotate_handler)
    return logger
