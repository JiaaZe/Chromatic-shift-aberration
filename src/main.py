from sys import (exit as sys_exit, argv as sys_argv)

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAbstractItemView, QTreeView, QListView, QLineEdit
from ui.MainWindow import Ui_MainWindow
from os.path import (exists as os_path_exists)
from os import (mkdir as os_mkdir)
import utils
from beads_pipeline import Correction

config_file = "config.ini"



def open_file_dialog(lineEdit: QLineEdit, mode=1, filetype_list=[], folder=""):
    """
    :param lineEdit:
    :param mode: 1. multiple directories, 2.single file, 3. multiple fils.
    :param filetype_list:
    :param folder: default open folder.
    :return:
    """
    fileDialog = QFileDialog()
    if len(folder) > 0:
        fileDialog.setDirectory(folder)
    path = ""
    if mode == 1:
        # multiple directories
        fileDialog.setFileMode(QFileDialog.Directory)
        # path = fileDialog.getExistingDirectory()
        fileDialog.setOption(QFileDialog.DontUseNativeDialog, True)
        fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_view = fileDialog.findChild(QListView)

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        f_tree_view = fileDialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if fileDialog.exec():
            path_list = fileDialog.selectedFiles()
            path = ';'.join(path_list)
    else:
        # single file
        fileDialog.setFileMode(QFileDialog.ExistingFile)
        name_filter = ""
        if len(filetype_list) > 0:
            for filetype in filetype_list:
                if len(name_filter) > 0:
                    name_filter += ";;"
                name_filter += "{} files (*.{} *.{})".format(filetype, filetype, filetype.upper())
            path = fileDialog.getOpenFileName(filter=name_filter)[0]
        else:
            path = fileDialog.getOpenFileName()[0]
    if mode == 3:
        cur_path = lineEdit.text()
        if len(path) > 0:
            if len(cur_path) > 0:
                lineEdit.setText(cur_path + ";" + path)
            else:
                lineEdit.setText(path)
    else:
        if len(path) > 0:
            lineEdit.setText(path)
    return path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.logger = utils.get_logger("/Logs")
        self.logger.info("\n==============================START==============================")
        self.ui = Ui_MainWindow()

        self.thread = None
        self.aberration = None

        self.ui.btn_start.connecte(self.start_pipeline)

    def start_pipeline(self):
        beads_red_path = self.ui.beads_red_path
        beads_green_path = self.ui.beads_green_path
        beads_blue_path = self.ui.beads_blue_path
        beads_csv_path = self.ui.beads_csv_path
        target_cm_path = self.ui.target_csv_path
        path_list = [beads_red_path, beads_green_path, beads_blue_path, beads_csv_path, target_cm_path]

        try:
            if self.thread is not None:
                self.thread.terminate()
            self.thread = QThread(self)
            self.logger.info('start doing stuff in: {}'.format(QThread.currentThread()))
            self.aberration = Correction(self.logger, path_list)
            self.aberration.moveToThread(self.thread)
            self.start_backgroung_work.connect(self.aberration.pipeline)

            self.aberration.append_text.connect(self.update_message)
            self.aberration.beads_finished.connect(self.show_vector_map)
            self.aberration.aberration_finished.connect(self.save_shifted_cm)

            self.thread.start()
            self.start_backgroung_work.emit()
        except Exception as e:
            self.logger.error("{}".format(e))
            self.ui.textbrowser_process.append("{}".format(e))
        else:
            msg = "Pipeline finished."
            self.logger.info(msg)
            self.ui.textbrowser_process.append(msg)

    def update_message(self, text):
        self.ui.textbrowser_process.append(text)

    def show_vector_map(self):
        ...

    def save_shifted_cm(self):
        ...


# Main access
if __name__ == '__main__':
    app = QApplication(sys_argv)
    window = MainWindow()
    window.show()
    sys_exit(app.exec())
