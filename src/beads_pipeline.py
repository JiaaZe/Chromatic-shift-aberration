import logging

from PyQt5.QtCore import (pyqtSignal as Signal, QObject)
from os.path import (split as os_path_split, join as os_path_join, exists as os_path_exists,
                     splitext as os_path_splitext)
from os import (listdir as os_listdir)
from beads_processing import train_beads, get_beads_df
from pandas import (read_csv as pd_read_csv, concat as pd_concat)
from numpy import (array as np_array, round as np_round)
from utils import write_csv


class Correction(QObject):
    append_text = Signal(str)
    update_progress = Signal(str)
    train_beads_finished = Signal(int)
    correction_finished = Signal(int)
    save_correction_finished = Signal(int)

    def __init__(self, logger: logging.Logger, path_list: list, lr_model: list, bgst_identifier: list):
        super().__init__()
        self.logger = logger
        self.beads_folder_list = path_list[0].split(";")
        self.beads_csv_path = path_list[1]
        self.target_csv_path_list = path_list[2].split(";")
        self.idenifier = bgst_identifier
        self.logger.info("Beads image folder path: {}".format(path_list[0]))
        self.logger.info("Beads csv path: {}".format(path_list[1]))
        self.logger.info("Target center of mass csv path: {}".format(path_list[2]))

        self.lr_model = lr_model
        self.img_shape = None
        self.beads_vector = None
        self.corrected_centerOfMass = None

    def train_model(self):
        if self.lr_model is not None:
            msg = "Reuse the beads model. Show the beads vector maps."
            self.logger.info(msg)
            self.append_text.emit(msg)
            return
        try:
            if len(self.beads_csv_path) > 0:
                lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, beads_df, pred_beads = train_beads(self.beads_csv_path,
                                                                                                 df_beads=None)
                self.lr_model = [lr_x_blue, lr_y_blue, lr_x_green, lr_y_green]
                self.beads_vector = [beads_df, pred_beads]
            else:
                beads_df = None
                num_beads_folders = len(self.beads_folder_list)
                for j, folder in enumerate(self.beads_folder_list):
                    file_list = os_listdir(folder)
                    beads_red_path = ""
                    beads_green_path = ""
                    beads_blue_path = ""
                    for i, filename in enumerate(file_list):
                        filename = filename.upper()
                        if not filename.endswith(".TIF"):
                            continue
                        if self.idenifier[0] in filename:
                            beads_red_path = os_path_join(folder, file_list[i])
                        elif self.idenifier[1] in filename:
                            beads_green_path = os_path_join(folder, file_list[i])
                        elif self.idenifier[2] in filename:
                            beads_blue_path = os_path_join(folder, file_list[i])
                    beads_tif_path_list = [beads_red_path, beads_green_path, beads_blue_path]
                    one_beads_df = get_beads_df(beads_tif_path_list, bgst=True)
                    beads_df = pd_concat([beads_df, one_beads_df], ignore_index=True)
                    finished = "==" * (j + 1)
                    left = ".." * (num_beads_folders - j - 1)
                    progress_text = "Process beads images: {}/{} [{}>{}] ".format(j + 1, num_beads_folders, finished, left)
                    if j == 0:
                        self.append_text.emit(progress_text)
                    else:
                        self.update_progress.emit(progress_text)

                self.append_text.emit("Start train chromatic shift model.")
                lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, _, pred_beads = train_beads(beads_path="",
                                                                                          df_beads=beads_df)
                self.lr_model = [lr_x_blue, lr_y_blue, lr_x_green, lr_y_green]
                self.beads_vector = [beads_df, pred_beads]
        except Exception as e:
            msg = "Error when train the beads model: {}.".format(e)
            raise Exception(msg)
            # self.logger.error("{}".format(e))
            # self.append_text.emit("Error: {}".format(e))
        else:
            msg = "Train the chromatic shift model sucessfully. Show the beads vector maps."
            self.logger.info(msg)
            self.append_text.emit(msg)

    def shift_centerOfMass(self):
        corrected_list = []
        lr_x_blue, lr_y_blue, lr_x_green, lr_y_green = self.lr_model
        try:
            for csv in self.target_csv_path_list:
                if len(csv) == 0:
                    continue
                df_target = pd_read_csv(csv, names=['red_x', 'red_y', 'green_x', 'green_y', 'blue_x', 'blue_y'],
                                        encoding="utf-8")
                r_centroid = np_array(df_target.loc[:, ['red_x', 'red_y']])
                g_centroid = np_array(df_target.loc[:, ['green_x', 'green_y']])
                b_centroid = np_array(df_target.loc[:, ['blue_x', 'blue_y']])

                pred_x_blue = lr_x_blue.predict(b_centroid)
                pred_y_blue = lr_y_blue.predict(b_centroid)

                pred_x_green = lr_x_green.predict(g_centroid)
                pred_y_green = lr_y_green.predict(g_centroid)

                g_new = np_round([g_centroid[:, 0] + pred_x_green, g_centroid[:, 1] + pred_y_green], 5).transpose()
                b_new = np_round([b_centroid[:, 0] + pred_x_blue, b_centroid[:, 1] + pred_y_blue], 5).transpose()

                corrected_list.append([r_centroid, g_new, b_new])
        except Exception as e:
            msg = "Error when correct the target center of mass: {}".format(e)
            raise Exception(msg)
            # self.logger.error("{}".format(e))
            # self.append_text.emit("Error: {}".format(e))
        else:
            msg = "Correct the center of mass sucessfully."
            self.logger.info(msg)
            self.append_text.emit(msg)
        return corrected_list

    def pipeline(self):
        try:
            # train beads model
            self.train_model()
            self.train_beads_finished.emit(1)

            if len(self.target_csv_path_list[0]) > 0:
                # correct taget center of mass csv
                self.corrected_centerOfMass = self.shift_centerOfMass()
                self.correction_finished.emit(1)

                # save corrected center of mass into csv
                self.save_corrected_centerOfMass()
                self.save_correction_finished.emit(1)
        except Exception as e:
            self.logger.error("{}".format(e))
            self.append_text.emit("{}".format(e))

    def get_beads_vector(self):
        return self.beads_vector

    def save_corrected_centerOfMass(self):
        try:
            for i, target_csv_path in enumerate(self.target_csv_path_list):
                corrected_com = self.corrected_centerOfMass[i]
                folder_name, fname = os_path_split(target_csv_path)
                corrected_path = os_path_join(folder_name, "corrected_" + fname)
                num_rep = 1
                prefix, suffix = os_path_splitext(corrected_path)
                while os_path_exists(corrected_path):
                    new_prefix = prefix + "_{}".format(num_rep)
                    corrected_path = new_prefix + suffix
                    num_rep += 1
                r, g, b = corrected_com
                write_csv(corrected_path, header=None,
                          data=np_array([r[:, 0], r[:, 1], g[:, 0], g[:, 1], b[:, 0], b[:, 1]]).transpose())
        except Exception as e:
            msg = "Error when save corrected center of mass csv file: {}.".format(e)
            raise Exception(msg)
            # self.logger.error(msg)
            # self.append_text.emit(msg)
        else:
            msg = "Save {} sucessfully.".format(corrected_path)
            self.logger.info(msg)
            self.append_text.emit(msg)
