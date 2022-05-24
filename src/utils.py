import logging
from logging.handlers import TimedRotatingFileHandler
from os.path import (exists as os_path_exists)
from os import (mkdir as os_mkdir)
from pandas import (DataFrame as pd_DataFrame)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import (pyqtSignal as Signal)


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


class ListBoxWidget(QListWidget):
    data_changed_signal = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.resize(312, 114)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

            links = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    links.append(str(url.toLocalFile()))
                else:
                    links.append(str(url.toString()))
            self.addItems(links)
            self.data_changed_signal.emit(1)
        else:
            event.ignore()

    def remove_items(self):
        selected = self.selectedItems()
        self.data_changed_signal.emit(1)
        for i in selected:
            row = self.row(i)
            self.takeItem(row)

    def get_all(self):
        self.selectAll()
        items = []
        for i in self.selectedItems():
            items.append(i.text())
        return items

    def clear(self):
        self.data_changed_signal.emit(1)
        super(ListBoxWidget, self).clear()

