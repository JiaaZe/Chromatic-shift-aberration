# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(910, 857)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.bead_group = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bead_group.sizePolicy().hasHeightForWidth())
        self.bead_group.setSizePolicy(sizePolicy)
        self.bead_group.setMinimumSize(QtCore.QSize(0, 70))
        self.bead_group.setMaximumSize(QtCore.QSize(16777215, 70))
        self.bead_group.setObjectName("bead_group")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.bead_group)
        self.gridLayout_5.setContentsMargins(-1, 4, -1, 4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(0, 3, 3, 3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.target_csv_path = QtWidgets.QLineEdit(self.bead_group)
        self.target_csv_path.setMinimumSize(QtCore.QSize(0, 22))
        self.target_csv_path.setMaximumSize(QtCore.QSize(16777215, 22))
        self.target_csv_path.setObjectName("target_csv_path")
        self.horizontalLayout_3.addWidget(self.target_csv_path)
        self.btn_target_csv_browser = QtWidgets.QPushButton(self.bead_group)
        self.btn_target_csv_browser.setMinimumSize(QtCore.QSize(0, 24))
        self.btn_target_csv_browser.setMaximumSize(QtCore.QSize(16777215, 24))
        self.btn_target_csv_browser.setObjectName("btn_target_csv_browser")
        self.horizontalLayout_3.addWidget(self.btn_target_csv_browser)
        self.btn_csv_clear = QtWidgets.QPushButton(self.bead_group)
        self.btn_csv_clear.setMinimumSize(QtCore.QSize(0, 24))
        self.btn_csv_clear.setMaximumSize(QtCore.QSize(16777215, 24))
        self.btn_csv_clear.setObjectName("btn_csv_clear")
        self.horizontalLayout_3.addWidget(self.btn_csv_clear)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.bead_group, 2, 0, 1, 1)
        self.frame_start = QtWidgets.QFrame(self.centralwidget)
        self.frame_start.setMinimumSize(QtCore.QSize(0, 100))
        self.frame_start.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_start.setObjectName("frame_start")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_start)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_start = QtWidgets.QPushButton(self.frame_start)
        self.btn_start.setMinimumSize(QtCore.QSize(0, 24))
        self.btn_start.setMaximumSize(QtCore.QSize(16777215, 24))
        self.btn_start.setObjectName("btn_start")
        self.horizontalLayout.addWidget(self.btn_start)
        self.textbrowser_process = QtWidgets.QTextBrowser(self.frame_start)
        self.textbrowser_process.setMinimumSize(QtCore.QSize(0, 50))
        self.textbrowser_process.setObjectName("textbrowser_process")
        self.horizontalLayout.addWidget(self.textbrowser_process)
        self.gridLayout_3.addWidget(self.frame_start, 3, 0, 1, 1)
        self.beads_image_group = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.beads_image_group.sizePolicy().hasHeightForWidth())
        self.beads_image_group.setSizePolicy(sizePolicy)
        self.beads_image_group.setMinimumSize(QtCore.QSize(0, 150))
        self.beads_image_group.setMaximumSize(QtCore.QSize(16777215, 150))
        self.beads_image_group.setObjectName("beads_image_group")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.beads_image_group)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.beads_image_group)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_beads_folders_browse = QtWidgets.QPushButton(self.beads_image_group)
        self.btn_beads_folders_browse.setObjectName("btn_beads_folders_browse")
        self.verticalLayout.addWidget(self.btn_beads_folders_browse)
        self.btn_beads_folders_remove = QtWidgets.QPushButton(self.beads_image_group)
        self.btn_beads_folders_remove.setObjectName("btn_beads_folders_remove")
        self.verticalLayout.addWidget(self.btn_beads_folders_remove)
        self.btn_beads_folders_clear = QtWidgets.QPushButton(self.beads_image_group)
        self.btn_beads_folders_clear.setObjectName("btn_beads_folders_clear")
        self.verticalLayout.addWidget(self.btn_beads_folders_clear)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.groupBox_3 = QtWidgets.QGroupBox(self.beads_image_group)
        self.groupBox_3.setMinimumSize(QtCore.QSize(210, 114))
        self.groupBox_3.setMaximumSize(QtCore.QSize(210, 114))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setMinimumSize(QtCore.QSize(70, 20))
        self.label_9.setMaximumSize(QtCore.QSize(70, 20))
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 0, 0, 1, 1)
        self.red_bgst_identifier = QtWidgets.QLineEdit(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.red_bgst_identifier.sizePolicy().hasHeightForWidth())
        self.red_bgst_identifier.setSizePolicy(sizePolicy)
        self.red_bgst_identifier.setMinimumSize(QtCore.QSize(0, 20))
        self.red_bgst_identifier.setMaximumSize(QtCore.QSize(16777215, 20))
        self.red_bgst_identifier.setObjectName("red_bgst_identifier")
        self.gridLayout_4.addWidget(self.red_bgst_identifier, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setMinimumSize(QtCore.QSize(70, 20))
        self.label_10.setMaximumSize(QtCore.QSize(70, 20))
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 1, 0, 1, 1)
        self.green_bgst_identifier = QtWidgets.QLineEdit(self.groupBox_3)
        self.green_bgst_identifier.setMinimumSize(QtCore.QSize(0, 20))
        self.green_bgst_identifier.setMaximumSize(QtCore.QSize(16777215, 20))
        self.green_bgst_identifier.setObjectName("green_bgst_identifier")
        self.gridLayout_4.addWidget(self.green_bgst_identifier, 1, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setMinimumSize(QtCore.QSize(70, 20))
        self.label_11.setMaximumSize(QtCore.QSize(70, 20))
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 2, 0, 1, 1)
        self.blue_bgst_identifier = QtWidgets.QLineEdit(self.groupBox_3)
        self.blue_bgst_identifier.setMinimumSize(QtCore.QSize(0, 20))
        self.blue_bgst_identifier.setMaximumSize(QtCore.QSize(16777215, 20))
        self.blue_bgst_identifier.setObjectName("blue_bgst_identifier")
        self.gridLayout_4.addWidget(self.blue_bgst_identifier, 2, 1, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_4.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.beads_image_group)
        self.groupBox_4.setMinimumSize(QtCore.QSize(141, 100))
        self.groupBox_4.setMaximumSize(QtCore.QSize(141, 100))
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_12.setContentsMargins(10, 10, 20, 10)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout_3.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout_3.setVerticalSpacing(10)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_13 = QtWidgets.QLabel(self.groupBox_4)
        self.label_13.setMinimumSize(QtCore.QSize(36, 22))
        self.label_13.setMaximumSize(QtCore.QSize(36, 22))
        self.label_13.setObjectName("label_13")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.image_height = QtWidgets.QLineEdit(self.groupBox_4)
        self.image_height.setMinimumSize(QtCore.QSize(60, 22))
        self.image_height.setMaximumSize(QtCore.QSize(60, 22))
        self.image_height.setObjectName("image_height")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.image_height)
        self.label_19 = QtWidgets.QLabel(self.groupBox_4)
        self.label_19.setMinimumSize(QtCore.QSize(36, 22))
        self.label_19.setMaximumSize(QtCore.QSize(36, 22))
        self.label_19.setObjectName("label_19")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.image_width = QtWidgets.QLineEdit(self.groupBox_4)
        self.image_width.setMinimumSize(QtCore.QSize(60, 22))
        self.image_width.setMaximumSize(QtCore.QSize(60, 22))
        self.image_width.setObjectName("image_width")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.image_width)
        self.gridLayout_12.addLayout(self.formLayout_3, 0, 0, 1, 1)
        self.horizontalLayout_4.addWidget(self.groupBox_4)
        self.gridLayout_3.addWidget(self.beads_image_group, 0, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setObjectName("splitter")
        self.chromatic_group = QtWidgets.QGroupBox(self.splitter)
        self.chromatic_group.setMinimumSize(QtCore.QSize(0, 0))
        self.chromatic_group.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.chromatic_group.setObjectName("chromatic_group")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.chromatic_group)
        self.gridLayout_10.setContentsMargins(3, 3, 3, 3)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.btn_export_beads = QtWidgets.QPushButton(self.chromatic_group)
        self.btn_export_beads.setObjectName("btn_export_beads")
        self.verticalLayout_10.addWidget(self.btn_export_beads)
        self.btn_save_beads_map = QtWidgets.QPushButton(self.chromatic_group)
        self.btn_save_beads_map.setObjectName("btn_save_beads_map")
        self.verticalLayout_10.addWidget(self.btn_save_beads_map)
        self.gridLayout_10.addLayout(self.verticalLayout_10, 0, 1, 1, 1)
        self.scroll_beads = QtWidgets.QScrollArea(self.chromatic_group)
        self.scroll_beads.setMinimumSize(QtCore.QSize(0, 100))
        self.scroll_beads.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.scroll_beads.setWidgetResizable(True)
        self.scroll_beads.setObjectName("scroll_beads")
        self.scroll_beads_content = QtWidgets.QWidget()
        self.scroll_beads_content.setGeometry(QtCore.QRect(0, 0, 799, 378))
        self.scroll_beads_content.setObjectName("scroll_beads_content")
        self.scroll_beads.setWidget(self.scroll_beads_content)
        self.gridLayout_10.addWidget(self.scroll_beads, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.splitter, 4, 0, 1, 1)
        self.group_beads_csv = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.group_beads_csv.sizePolicy().hasHeightForWidth())
        self.group_beads_csv.setSizePolicy(sizePolicy)
        self.group_beads_csv.setMinimumSize(QtCore.QSize(0, 70))
        self.group_beads_csv.setMaximumSize(QtCore.QSize(16777215, 70))
        self.group_beads_csv.setObjectName("group_beads_csv")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.group_beads_csv)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 3, 3, 3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.group_beads_csv)
        self.label_7.setMinimumSize(QtCore.QSize(0, 28))
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 28))
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.beads_csv_path = QtWidgets.QLineEdit(self.group_beads_csv)
        self.beads_csv_path.setMinimumSize(QtCore.QSize(0, 22))
        self.beads_csv_path.setMaximumSize(QtCore.QSize(16777215, 22))
        self.beads_csv_path.setInputMask("")
        self.beads_csv_path.setText("")
        self.beads_csv_path.setDragEnabled(False)
        self.beads_csv_path.setObjectName("beads_csv_path")
        self.horizontalLayout_2.addWidget(self.beads_csv_path)
        self.btn_beadscsv_browse = QtWidgets.QPushButton(self.group_beads_csv)
        self.btn_beadscsv_browse.setMinimumSize(QtCore.QSize(0, 24))
        self.btn_beadscsv_browse.setMaximumSize(QtCore.QSize(16777215, 24))
        self.btn_beadscsv_browse.setObjectName("btn_beadscsv_browse")
        self.horizontalLayout_2.addWidget(self.btn_beadscsv_browse)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.group_beads_csv, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Beads shift aberration"))
        self.bead_group.setTitle(_translate("MainWindow", "Center of mass csv files"))
        self.target_csv_path.setPlaceholderText(_translate("MainWindow", "Select paths"))
        self.btn_target_csv_browser.setText(_translate("MainWindow", "Browse"))
        self.btn_csv_clear.setText(_translate("MainWindow", "Clear"))
        self.btn_start.setText(_translate("MainWindow", "START"))
        self.beads_image_group.setTitle(_translate("MainWindow", "Beads Images"))
        self.label_5.setText(_translate("MainWindow", "Beads Image Folders"))
        self.btn_beads_folders_browse.setText(_translate("MainWindow", "Browse"))
        self.btn_beads_folders_remove.setText(_translate("MainWindow", "Remove"))
        self.btn_beads_folders_clear.setText(_translate("MainWindow", "Clear"))
        self.groupBox_3.setTitle(_translate("MainWindow", "BGST Channel Identifier"))
        self.label_9.setText(_translate("MainWindow", "RED-BGST"))
        self.red_bgst_identifier.setText(_translate("MainWindow", "CY3-BGST"))
        self.label_10.setText(_translate("MainWindow", "GREEN-BGST"))
        self.green_bgst_identifier.setText(_translate("MainWindow", "GFP-BGST"))
        self.label_11.setText(_translate("MainWindow", "BLUE-BGST"))
        self.blue_bgst_identifier.setText(_translate("MainWindow", "CY5-BGST"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Image Size"))
        self.label_13.setText(_translate("MainWindow", "Height"))
        self.image_height.setText(_translate("MainWindow", "1080"))
        self.label_19.setText(_translate("MainWindow", "Width"))
        self.image_width.setText(_translate("MainWindow", "1280"))
        self.chromatic_group.setTitle(_translate("MainWindow", "Chromatic shift calibration"))
        self.btn_export_beads.setText(_translate("MainWindow", "Export"))
        self.btn_save_beads_map.setText(_translate("MainWindow", "Save"))
        self.group_beads_csv.setTitle(_translate("MainWindow", "Beads CSV"))
        self.label_7.setText(_translate("MainWindow", "Beads CSV file "))
        self.beads_csv_path.setPlaceholderText(_translate("MainWindow", "Select beads csv file path"))
        self.btn_beadscsv_browse.setText(_translate("MainWindow", "Browse"))
