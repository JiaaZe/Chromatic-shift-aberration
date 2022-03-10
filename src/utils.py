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


def read_tif(path, IMG_HEIGHT, IMG_WIDTH):
    img = cv2_imread(path, -1)
    h, w = img.shape
    h_expand = IMG_HEIGHT - h
    w_expand = IMG_WIDTH - w
    if h_expand + w_expand > 0:
        new_img = np_zeros((IMG_HEIGHT, IMG_WIDTH))
        new_img[0:h, 0:w] = img
        new_img[h + h_expand:, :w] = img[h - h_expand:h, :][::-1, :]
        new_img[:, -w_expand:] = new_img[:, -w_expand * 2:-w_expand][:, ::-1]
        img = new_img
    return img


def write_csv():
    ...
