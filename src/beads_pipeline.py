import logging

from PyQt5.QtCore import (pyqtSignal as Signal, QObject, QThread)
from os.path import (split as os_path_split, join as os_path_join, exists as os_path_exists,
                     splitext as os_path_splitext)
from src.beads_processing import train_beads, process_bead
from pandas import (DataFrame as pd_DataFrame, read_csv as pd_read_csv)
from numpy import (array as np_array, round as np_round)
from utils import write_csv


class Correction(QObject):
    append_text = Signal(str)
    train_beads_finished = Signal(int)
    correction_finished = Signal(int)
    save_correction_finished = Signal(int)

    def __init__(self, logger: logging.Logger, path_list: list, lr_model: list):
        super().__init__()
        self.logger = logger
        self.beads_red_path = path_list[0]
        self.beads_green_path = path_list[1]
        self.beads_blue_path = path_list[2]
        self.beads_csv_path = path_list[3]
        self.target_csv_path_list = path_list[4].split(";")
        self.logger.info("Beads red path: {}".format(self.beads_red_path))
        self.logger.info("Beads green path: {}".format(self.beads_green_path))
        self.logger.info("Beads blue path: {}".format(self.beads_blue_path))
        self.logger.info("Target center of mass csv path: {}".format(path_list[4]))

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
                lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, beads_df, pred_beads = train_beads(self.beads_csv_path)
                self.lr_model = [lr_x_blue, lr_y_blue, lr_x_green, lr_y_green]
                self.beads_vector = [beads_df, pred_beads]
            else:
                beads_tif_path_list = [self.beads_red_path, self.beads_green_path, self.beads_blue_path]
                lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, beads_df, pred_beads, img_shape = process_bead(
                    beads_tif_path_list, False)
                self.img_shape = img_shape
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
                df_target = pd_read_csv(csv, names=['red_y', 'red_x', 'green_y', 'green_x', 'blue_y', 'blue_x'])
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
                          data=np_array([r[:, 1], r[:, 0], g[:, 1], g[:, 0], b[:, 1], b[:, 0]]).transpose())
        except Exception as e:
            msg = "Error when save corrected center of mass csv file: {}.".format(e)
            raise Exception(msg)
            # self.logger.error(msg)
            # self.append_text.emit(msg)
        else:
            msg = "Save {} sucessfully.".format(corrected_path)
            self.logger.info(msg)
            self.append_text.emit(msg)