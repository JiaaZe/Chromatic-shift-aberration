import logging
from logging.handlers import TimedRotatingFileHandler
from os.path import (exists as os_path_exists)
from os import (mkdir as os_mkdir)
from pandas import (DataFrame as pd_DataFrame)


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_name = log_path + "/logFile" + '.log'
    logfile = log_name
    if not os_path_exists(log_path):
        os_mkdir(log_path)

    file_handler = TimedRotatingFileHandler(logfile, when='H', interval=3, backupCount=4, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)

    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    console_logger.setFormatter(formatter)

    logger.addHandler(console_logger)
    logger.addHandler(file_handler)

    return logger


def write_csv(path, data, header):
    if header is not None and len(header) > 0 and len(header) != data.shape[1]:
        raise Exception("Header length incompatible with data length.")
    df = pd_DataFrame(columns=header, data=data)
    df.to_csv(path, index=False, header=False, encoding="utf-8")
