import time
from sys import (exit as sys_exit, argv as sys_argv)
from configparser import ConfigParser as configparser_ConfigParser
from os.path import (exists as os_path_exists)
from os import (listdir as os_listdir)

from ui.MainWindow import Ui_MainWindow
from beads_pipeline import Correction
import utils

from PyQt5 import QtWidgets
from PyQt5.QtCore import (QThread, pyqtSignal as Signal)
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAbstractItemView, QTreeView, QListView, QLineEdit, \
    QVBoxLayout, QWidget
from pandas import (DataFrame as pd_DataFrame)
from numpy import (arange as np_arange)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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
        # fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
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
    start_backgroung_work = Signal()

    def __init__(self):
        super().__init__()

        self.logger = utils.get_logger("./Logs")
        self.logger.info("\n==============================START==============================")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.cfg = configparser_ConfigParser()
        if os_path_exists(config_file):
            self.cfg.read(config_file)
            self.set_params_from_cfg()
        else:
            self.cfg['parameters'] = {}

        self.thread = None
        self.correction = None
        self.old_path_list = None
        self.old_lr_model = None
        self.old_beads_vector = None
        self.reuse = True
        self.img_shape_changed = False

        self.ui.btn_start.clicked.connect(self.start_pipeline)
        self.ui.btn_beads_folders_browse.clicked.connect(self.open_beads_folder)
        self.ui.btn_beadscsv_browse.clicked.connect(self.open_beads_csv_file)
        self.ui.btn_target_csv_browser.clicked.connect(self.open_target_file)

        self.ui.btn_csv_clear.clicked.connect(self.clear_target_csv_path)
        self.ui.btn_save_beads_map.clicked.connect(self.save_beads_map)

        # text change
        self.ui.beads_folder_path.textChanged.connect(self.handle_path_changed)
        self.ui.beads_csv_path.textChanged.connect(self.handle_path_changed)
        self.ui.image_width.textChanged.connect(self.handle_img_shape_changed)
        self.ui.image_height.textChanged.connect(self.handle_img_shape_changed)

        self.beads_vector_maps = None

    def set_params_from_cfg(self):
        self.ui.red_bgst_identifier.setText(self.cfg.get("parameters", "red_bgst_identifier"))
        self.ui.green_bgst_identifier.setText(self.cfg.get("parameters", "green_bgst_identifier"))
        self.ui.blue_bgst_identifier.setText(self.cfg.get("parameters", "blue_bgst_identifier"))

    def handle_path_changed(self):
        self.ui.textbrowser_process.clear()
        self.reuse = False

    def handle_img_shape_changed(self):
        self.img_shape_changed = True

    def start_pipeline(self):
        self.ui.btn_start.setDisabled(True)
        try:
            msg = "[{}]===========Start pipeline============".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.logger.info(msg)
            self.update_message(msg)

            # beads_red_path = self.ui.beads_red_path.text()
            # beads_green_path = self.ui.beads_green_path.text()
            # beads_blue_path = self.ui.beads_blue_path.text()
            beads_folders_path = self.ui.beads_folder_path.toPlainText()
            beads_csv_path = self.ui.beads_csv_path.text()
            target_cm_path = self.ui.target_csv_path.text()
            if len(beads_csv_path) > 0:
                msg = "Using beads csv file to train the model."
                self.logger.info(msg)
                self.update_message(msg)
            elif len(beads_folders_path) > 0:
                msg = "Using beads images to train the model."
                self.logger.info(msg)
                self.update_message(msg)
            else:
                msg = "Invalid beads path."
                raise Exception(msg)
            if len(target_cm_path) == 0:
                msg = "Please select target center of mass csv path."
                raise Exception(msg)

            shape_msg = ""
            if len(self.ui.image_width.text()) < 0:
                shape_msg += "Please input beads image width."
            if len(self.ui.image_height.text()) < 0:
                shape_msg += "Please input beads image width."
            if len(shape_msg) > 0:
                raise Exception(shape_msg)
            identifier_list = self.check_identifier()
            self.cfg['parameters']['red_bgst_identifier'] = identifier_list[0]
            self.cfg['parameters']['green_bgst_identifier'] = identifier_list[1]
            self.cfg['parameters']['blue_bgst_identifier'] = identifier_list[2]
            with open(config_file, 'w') as configfile:
                self.cfg.write(configfile)
            # path_list = [beads_red_path, beads_green_path, beads_blue_path, beads_csv_path, target_cm_path]
            path_list = [beads_folders_path, beads_csv_path, target_cm_path]

            # reuse the old beads lr model
            if self.reuse:
                ...
            else:
                self.old_lr_model = None
                self.old_beads_vector = None

                self.ui.scroll_beads_content = QtWidgets.QWidget()
                self.ui.scroll_beads.setWidget(self.ui.scroll_beads_content)
                self.ui.btn_save_beads_map.setDisabled(True)
            if self.img_shape_changed:
                self.ui.scroll_beads_content = QtWidgets.QWidget()
                self.ui.scroll_beads.setWidget(self.ui.scroll_beads_content)
                self.ui.btn_save_beads_map.setDisabled(True)

            if self.thread is not None:
                self.thread.terminate()
            self.thread = QThread(self)
            self.logger.info('start doing stuff in: {}'.format(QThread.currentThread()))
            self.correction = Correction(self.logger, path_list, self.old_lr_model, identifier_list)
            self.correction.moveToThread(self.thread)
            self.start_backgroung_work.connect(self.correction.pipeline)

            self.correction.append_text.connect(self.update_message)
            self.correction.train_beads_finished.connect(self.show_vector_map)
            # self.correction.correction_finished.connect(self.save_shifted_centerOfMass)

            self.thread.start()
            self.start_backgroung_work.emit()

            self.old_path_list = path_list
        except Exception as e:
            self.ui.btn_start.setEnabled(True)
            self.logger.error("{}".format(e))
            self.update_message("{}".format(e))

    def update_message(self, text):
        self.ui.textbrowser_process.append(text)

    def open_beads_folder(self):
        open_file_dialog(self.ui.beads_folder_path, mode=1)

    def open_beads_csv_file(self):
        open_file_dialog(self.ui.beads_csv_path, mode=2, filetype_list=["csv", "xlsx", "xls"])

    def open_target_file(self):
        open_file_dialog(self.ui.target_csv_path, mode=3, filetype_list=["csv", "xlsx", "xls"])

    def clear_target_csv_path(self):
        self.ui.target_csv_path.clear()

    def check_identifier(self):
        # Check identifier for each image folder
        err_msg = ""
        bgst_identifier = [self.ui.red_bgst_identifier.text().upper(), self.ui.green_bgst_identifier.text().upper(),
                           self.ui.blue_bgst_identifier.text().upper()]
        folder_list = self.ui.beads_folder_path.toPlainText().split(";")
        for folder in folder_list:
            bgst_identifier_flag = [0, 0, 0]
            file_list = os_listdir(folder)
            for file_name in file_list:
                file_name = file_name.upper()
                if not file_name.endswith(".TIF"):
                    continue
                for i in range(3):
                    if bgst_identifier[i] in file_name \
                            and bgst_identifier[(i - 2) % 3] not in file_name \
                            and bgst_identifier[(i - 1) % 3] not in file_name:
                        bgst_identifier_flag[i] = bgst_identifier_flag[i] + 1
            if sum(bgst_identifier_flag) != 3:
                err_msg += "BGST identifier can not identify channel images in the folder: {}\n".format(folder)
        if len(err_msg) > 0:
            raise Exception(err_msg)
        return bgst_identifier

    def show_vector_map(self):
        # get lr model list
        self.old_lr_model = self.correction.lr_model
        if self.reuse:
            if not self.img_shape_changed:
                return
            else:
                beads_df, pred_beads = self.old_beads_vector
        else:
            self.reuse = True
            self.img_shape_changed = False
            beads_df, pred_beads = self.correction.get_beads_vector()
            self.old_beads_vector = [beads_df, pred_beads]
        img_shape = [int(self.ui.image_width.text()), int(self.ui.image_height.text())]
        arrow_df = pd_DataFrame({'diff_green_y': 100 * (beads_df['green_y'] - beads_df['red_y']),
                                 'diff_green_x': 100 * (beads_df['green_x'] - beads_df['red_x']),
                                 'diff_blue_y': 100 * (beads_df['blue_y'] - beads_df['red_y']),
                                 'diff_blue_x': 100 * (beads_df['blue_x'] - beads_df['red_x']),
                                 'pred_diff_green_y': 100 * (pred_beads['green_y'] - beads_df['red_y']),
                                 'pred_diff_green_x': 100 * (pred_beads['green_x'] - beads_df['red_x']),
                                 'pred_diff_blue_y': 100 * (pred_beads['blue_y'] - beads_df['red_y']),
                                 'pred_diff_blue_x': 100 * (pred_beads['blue_x'] - beads_df['red_x'])})

        qScrollLayout = QVBoxLayout(self.ui.scroll_beads_content)
        qfigWidget = QWidget(self.ui.scroll_beads_content)

        static_canvas = FigureCanvas(Figure(figsize=(7, 7)))
        text_size = 6
        r = static_canvas.get_renderer()

        subplot_axes = static_canvas.figure.subplots(2, 2)
        # static_canvas.figure.tight_layout()
        # static_canvas.figure.subplots_adjust(hspace=0.3)
        # 0,0 original arrow
        subplot_axes[0, 0].title.set_text("original arrow")
        subplot_axes[0, 0].title.set_size(10)

        x_lim = (min(beads_df["red_x"].min(), (beads_df["red_x"] + arrow_df['diff_green_x']).min(),
                     (beads_df["red_x"] + arrow_df['diff_blue_x']).min()) - 10,
                 max(beads_df["red_x"].max(), (beads_df["red_x"] + arrow_df['diff_green_x']).max(),
                     (beads_df["red_x"] + arrow_df['diff_blue_x']).max()) + 10)
        y_lim = (min(beads_df["red_y"].min(), (beads_df["red_y"] + arrow_df['diff_green_y']).min(),
                     (beads_df["red_y"] + arrow_df['diff_blue_y']).min()) - 10,
                 max(beads_df["red_y"].max(), (beads_df["red_y"] + arrow_df['diff_green_y']).max(),
                     (beads_df["red_y"] + arrow_df['diff_blue_y']).max()) + 10)
        subplot_axes[0, 0].set_xlim(x_lim)

        subplot_axes[0, 0].set_ylim(y_lim)

        subplot_axes[0, 1].set_xlim(x_lim)

        subplot_axes[0, 1].set_ylim(y_lim)

        for index, row in beads_df.iterrows():
            subplot_axes[0, 0].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["diff_green_x"],
                                     arrow_df.iloc[index]["diff_green_y"], color='green', head_width=6, lw=0.4)
            subplot_axes[0, 0].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["diff_blue_x"],
                                     arrow_df.iloc[index]["diff_blue_y"], color='blue', head_width=6, lw=0.4)
        subplot_axes[0, 0].add_patch(Rectangle((0, 0), img_shape[0], img_shape[1], fill=False, linewidth=1))
        subplot_axes[0, 0].axis('off')

        # 0,1 shifted arrow
        subplot_axes[0, 1].title.set_text("shifted arrow")
        subplot_axes[0, 1].title.set_size(10)
        for index, row in pred_beads.iterrows():
            subplot_axes[0, 1].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["pred_diff_green_x"],
                                     arrow_df.iloc[index]["pred_diff_green_y"], color='green', head_width=6, lw=0.4)
            subplot_axes[0, 1].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["pred_diff_blue_x"],
                                     arrow_df.iloc[index]["pred_diff_blue_y"], color='blue', head_width=6, lw=0.4)
        subplot_axes[0, 1].add_patch(Rectangle((0, 0), img_shape[0], img_shape[1], fill=False, linewidth=1))
        subplot_axes[0, 1].axis('off')

        # scattor
        scatter_df = pd_DataFrame(
            {'green_x': beads_df['green_x'] - beads_df['red_x'], 'green_y': beads_df['green_y'] - beads_df['red_y'],
             'blue_x': beads_df['blue_x'] - beads_df['red_x'], 'blue_y': beads_df['blue_y'] - beads_df['red_y'],
             'pred_green_x': pred_beads['green_x'] - pred_beads['red_x'],
             'pred_green_y': pred_beads['green_y'] - pred_beads['red_y'],
             'pred_blue_x': pred_beads['blue_x'] - pred_beads['red_x'],
             'pred_blue_y': pred_beads['blue_y'] - pred_beads['red_y']})
        scatter_df = scatter_df.apply(lambda x: 67 * x)

        # 1,0 original related
        subplot_axes[1, 0].title.set_text("original related")
        subplot_axes[1, 0].title.set_size(10)
        x_max = max(abs(min(scatter_df[["green_x", "blue_x"]].min())), max(scatter_df[["green_x", "blue_x"]].max()))
        y_max = max(abs(min(scatter_df[["green_y", "blue_y"]].min())), max(scatter_df[["green_y", "blue_y"]].max()))
        lim = max(x_max, y_max)
        if lim % 10 != 0:
            lim = (int(lim / 10) + 1) * 10
        xy_lim = (-lim, lim)
        xy_ticks = np_arange(-lim, lim + 10, 10)
        xy_labels = [xy_ticks[i] if i % 2 == 0 else " " for i in range(len(xy_ticks))]
        subplot_axes[1, 0].set_xlim(xy_lim)
        subplot_axes[1, 0].set_ylim(xy_lim)
        subplot_axes[1, 0].set_xticks(xy_ticks, xy_labels)
        subplot_axes[1, 0].set_yticks(xy_ticks, xy_labels)
        subplot_axes[1, 0].scatter(scatter_df["green_x"], scatter_df["green_y"], c='g', s=5, alpha=0.4)
        subplot_axes[1, 0].scatter(scatter_df["blue_x"], scatter_df["blue_y"], c='b', s=5, alpha=0.4)
        subplot_axes[1, 0].axvline(c="black", lw=1)
        subplot_axes[1, 0].axhline(c="black", lw=1)
        subplot_axes[1, 0].spines['top'].set_visible(False)
        subplot_axes[1, 0].spines['right'].set_visible(False)
        subplot_axes[1, 0].set_xlabel("x position of center (nm)")
        subplot_axes[1, 0].set_ylabel("y position of center (nm)")

        # box information
        title_text = subplot_axes[1, 0].text(lim, lim, "mean±SD", ha="right", va="top", ma="center",
                                             size=text_size)
        bb_title = title_text.get_window_extent(renderer=r)
        bb_title = bb_title.transformed(subplot_axes[1, 0].transData.inverted())

        text_height = bb_title.ymax - bb_title.ymin

        green_text = subplot_axes[1, 0].text(bb_title.x1, bb_title.y0 - text_height / 2,
                                             "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(
                                                 scatter_df['green_x'].mean(),
                                                 scatter_df['green_x'].std(),
                                                 scatter_df['green_y'].mean(),
                                                 scatter_df['green_y'].std()),
                                             ha="right", va="top", ma="left", size=text_size, c='g')
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 0].transData.inverted())
        green_text.set_ha("left")
        green_width = bb_green.xmax - bb_green.xmin

        blue_text = subplot_axes[1, 0].text(bb_green.x0, bb_green.y0,
                                            "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(
                                                scatter_df['blue_x'].mean(),
                                                scatter_df['blue_x'].std(),
                                                scatter_df['blue_y'].mean(),
                                                scatter_df['blue_y'].std()),
                                            ha="left", va="top", ma="left", size=text_size, c='b')
        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 0].transData.inverted())
        blue_width = bb_blue.xmax - bb_blue.xmin

        max_width = max(blue_width, green_width)

        if blue_width > green_width:
            shift = bb_blue.x1 - bb_green.x1
        else:
            shift = 0

        new_x0 = bb_blue.x0 - shift - text_height / 2
        blue_text.set_x(new_x0)
        green_text.set_x(new_x0)

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 0].transData.inverted())
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 0].transData.inverted())

        title_shift = bb_title.x0 - text_height / 2 - bb_blue.x0
        title_text.set_x(bb_title.x1 - title_shift / 2)
        title_text.set_y(bb_title.y1 - text_height / 2)

        count_text = subplot_axes[1, 0].text(bb_blue.x1, bb_blue.y0,
                                             "n={}".format(len(scatter_df)), ha="right", va="top",
                                             ma="center",
                                             size=text_size)
        bb_count = count_text.get_window_extent(renderer=r)
        bb_count = bb_count.transformed(subplot_axes[1, 0].transData.inverted())

        count_shift = bb_count.x0 - bb_blue.x0
        count_text.set_x(bb_count.x1 - count_shift / 2)

        bbox_shift = text_height / 3
        bbox_xy = (bb_blue.x0 - text_height / 3, bb_count.y0)
        bbox_width = max_width + 2 * bbox_shift
        bbox_height = bb_title.y1 - bb_count.y0 - bbox_shift / 2

        subplot_axes[1, 0].add_patch(
            Rectangle(bbox_xy, bbox_width, bbox_height, fc=(1, 1, 1), ec=(0, 0, 0), lw=0.5, alpha=0.5))

        # 1,1 shifted related
        subplot_axes[1, 1].title.set_text("shifted related")
        subplot_axes[1, 1].title.set_size(10)
        subplot_axes[1, 1].set_xlim(xy_lim)
        subplot_axes[1, 1].set_ylim(xy_lim)
        subplot_axes[1, 1].set_xticks(xy_ticks, xy_labels)
        subplot_axes[1, 1].set_yticks(xy_ticks, xy_labels)

        subplot_axes[1, 1].scatter(scatter_df["pred_green_x"], scatter_df["pred_green_y"], c='g', s=5, alpha=0.4)
        subplot_axes[1, 1].scatter(scatter_df["pred_blue_x"], scatter_df["pred_blue_y"], c='b', s=5, alpha=0.4)

        subplot_axes[1, 1].axvline(c="black", lw=1)
        subplot_axes[1, 1].axhline(c="black", lw=1)
        subplot_axes[1, 1].spines['top'].set_visible(False)
        subplot_axes[1, 1].spines['right'].set_visible(False)
        subplot_axes[1, 1].set_xlabel("x position of center (nm)")
        subplot_axes[1, 1].set_ylabel("y position of center (nm)")

        # box information
        title_text = subplot_axes[1, 1].text(lim, lim, "mean±SD", ha="right", va="top", ma="center",
                                             size=text_size)
        bb_title = title_text.get_window_extent(renderer=r)
        bb_title = bb_title.transformed(subplot_axes[1, 1].transData.inverted())

        green_text = subplot_axes[1, 1].text(bb_title.x1, bb_title.y0 - text_height / 2,
                                             "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(
                                                 scatter_df['pred_green_x'].mean(),
                                                 scatter_df['pred_green_x'].std(),
                                                 scatter_df['pred_green_y'].mean(),
                                                 scatter_df['pred_green_y'].std()),
                                             ha="right", va="top", ma="left", size=text_size, c='g')
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 1].transData.inverted())
        green_text.set_ha("left")
        green_width = bb_green.xmax - bb_green.xmin

        blue_text = subplot_axes[1, 1].text(bb_green.x0, bb_green.y0,
                                            "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(
                                                scatter_df['pred_blue_x'].mean(),
                                                scatter_df['pred_blue_x'].std(),
                                                scatter_df['pred_blue_y'].mean(),
                                                scatter_df['pred_blue_y'].std()),
                                            ha="left", va="top", ma="left", size=text_size, c='b')

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 1].transData.inverted())
        blue_width = bb_blue.xmax - bb_blue.xmin

        if blue_width > green_width:
            shift = bb_blue.x1 - bb_green.x1
        else:
            shift = 0

        new_x0 = bb_blue.x0 - shift - text_height / 2
        blue_text.set_x(new_x0)
        green_text.set_x(new_x0)

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 1].transData.inverted())
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 1].transData.inverted())

        max_width = max(blue_width, green_width)

        title_shift = bb_title.x0 - text_height / 2 - bb_blue.x0
        title_text.set_x(bb_title.x1 - title_shift / 2)
        title_text.set_y(bb_title.y1 - text_height / 2)

        count_text = subplot_axes[1, 1].text(bb_blue.x1, bb_blue.y0,
                                             "n={}".format(len(scatter_df)), ha="right", va="top",
                                             ma="center",
                                             size=text_size)
        bb_count = count_text.get_window_extent(renderer=r)
        bb_count = bb_count.transformed(subplot_axes[1, 1].transData.inverted())

        count_shift = bb_count.x0 - bb_blue.x0
        count_text.set_x(bb_count.x1 - count_shift / 2)

        bbox_shift = text_height / 3
        bbox_xy = (bb_blue.x0 - text_height / 3, bb_count.y0)
        bbox_width = max_width + 2 * bbox_shift
        bbox_height = bb_title.y1 - bb_count.y0 - bbox_shift / 2

        subplot_axes[1, 1].add_patch(
            Rectangle(bbox_xy, bbox_width, bbox_height, fc=(1, 1, 1), ec=(0, 0, 0), lw=0.5, alpha=0.5))

        subplot_axes[0, 0].set_aspect(1)
        subplot_axes[0, 1].set_aspect(1)
        subplot_axes[1, 0].set_aspect(1)
        subplot_axes[1, 1].set_aspect(1)
        plotLayout = QVBoxLayout()
        self.beads_vector_maps = static_canvas
        plotLayout.addWidget(static_canvas)
        qfigWidget.setLayout(plotLayout)

        static_canvas.setMinimumSize(static_canvas.size())

        qScrollLayout.addWidget(qfigWidget)
        self.ui.scroll_beads_content.setLayout(qScrollLayout)

        self.ui.scroll_beads_content.show()
        self.ui.btn_save_beads_map.setEnabled(True)
        self.ui.btn_start.setEnabled(True)

    def save_beads_map(self):
        try:
            save_path, save_type = QFileDialog.getSaveFileName(self, "Save File", "./beads vector maps",
                                                               'pdf (*.pdf);; png (*.png);;jpg (*.jpg)')
            if save_type == "pdf (*.pdf)":
                with PdfPages(save_path) as pdf:
                    pdf.savefig(self.beads_vector_maps.figure, dpi=120)
            else:
                self.beads_maps.figure.savefig(save_path)
        except Exception as e:
            msg = "Error when save the beads vector maps: {}".format(e)
            self.logger.error(e)
            self.update_message(msg)
        else:
            msg = "Beads vector maps saved as {}".format(save_path)
            self.logger.info(msg)
            self.update_message(msg)


# Main access
if __name__ == '__main__':
    app = QApplication(sys_argv)
    window = MainWindow()
    window.show()
    sys_exit(app.exec())
