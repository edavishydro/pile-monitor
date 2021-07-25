# -*- coding: utf-8 -*-
from tkinter import W
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import pandas as pd
import glob

# import os
from collections import defaultdict
import pile_funcs
from pathlib import Path


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Pile Monitor")
        MainWindow.resize(800, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        self.summaryTable = QtWidgets.QTableView(self.centralwidget)
        self.summaryTable.setAlternatingRowColors(True)
        self.summaryTable.setGeometry(QtCore.QRect(20, 50, 761, 231))
        self.summaryTable.setWordWrap(True)
        self.summaryTable.setObjectName("summaryTable")
        self.summaryTable.horizontalHeader().setDefaultSectionSize(125)
        self.summaryTable.setColumnWidth(0, 200)
        self.summaryTable.verticalHeader().setCascadingSectionResizes(True)

        self.strikeTable = QtWidgets.QTableView(self.centralwidget)
        self.strikeTable.setAlternatingRowColors(True)
        self.strikeTable.setGeometry(QtCore.QRect(20, 310, 761, 380))
        self.strikeTable.setObjectName("strikeTable")
        self.strikeTable.setSizePolicy(sizePolicy)
        self.strikeTable.setWordWrap(False)
        self.strikeTable.horizontalHeader().setCascadingSectionResizes(True)
        self.strikeTable.horizontalHeader().setDefaultSectionSize(125)
        self.strikeTable.verticalHeader().setCascadingSectionResizes(True)

        MainWindow.setCentralWidget(self.centralwidget)

        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)

        self.summLabel = QtWidgets.QLabel(self.centralwidget)
        self.summLabel.setGeometry(QtCore.QRect(20, 30, 200, 16))
        self.summLabel.setFont(font)
        self.summLabel.setTextFormat(QtCore.Qt.PlainText)
        self.summLabel.setObjectName("summLabel")

        self.strkLabel = QtWidgets.QLabel(self.centralwidget)
        self.strkLabel.setGeometry(QtCore.QRect(20, 290, 200, 16))
        self.strkLabel.setFont(font)
        self.strkLabel.setTextFormat(QtCore.Qt.PlainText)
        self.strkLabel.setObjectName("strkLabel")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(522, 10, 261, 32))
        self.pushButton.setObjectName("pushButton")

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.message = "Please select processing directory..."
        self.statusbar.showMessage(self.message)

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.setOutputDir)
        self.outputDir = ""
        self.summ_df = pd.DataFrame()
        self.strk_df = pd.DataFrame()
        self.strk_raw = pd.DataFrame()
        self.run_dict = dict()
        self.xlsx_name = ""

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pile Monitor"))
        self.summLabel.setText(_translate("MainWindow", "Summary Strikes Report"))
        self.strkLabel.setText(_translate("MainWindow", "Peaks Report"))
        self.pushButton.setText(
            _translate("MainWindow", "Choose processing directory...")
        )

    def populate_data(self, dir1):
        self.alr_proc = list()
        self.run_dict, self.strk_raw, strikes, xlsx_name = pile_funcs.process_dir(
            dir1, self.alr_proc
        )
        # print(self.strk_raw)
        self.summ_df, self.run_df = pile_funcs.summaries(self.strk_raw, self.run_dict)
        self.strk_df = strikes
        self.xlsx_name = xlsx_name
        strikes = self.strk_df.reset_index().sort_values(axis=0, by=["filename", "ix"])
        self.summaryTable.setModel(pandasModel(self.summ_df))
        self.strikeTable.setModel(pandasModel(strikes))
        pile_funcs.output_excel(self.strk_df, self.summ_df, self.run_df, xlsx_name)
        self.alr_proc = [f for f in glob.glob("*.wav")]

    def repopulate_data(self):
        run_set, strikes_raw, strikes, xlsx_name = pile_funcs.process_dir(
            self.outputDir, self.alr_proc
        )
        self.run_dict.update(run_set)
        # print(self.strk_raw)
        strikes_raw_2 = strikes_raw.reset_index()
        self.strk_raw = self.strk_raw.append(strikes_raw_2).set_index("filename")
        self.summ_df, self.run_df = pile_funcs.summaries(self.strk_raw, self.run_dict)
        self.strk_df = self.strk_df.append(strikes)
        strikes = self.strk_df.reset_index().sort_values(axis=0, by=["filename", "ix"])
        self.summaryTable.setModel(pandasModel(self.summ_df))
        self.strikeTable.setModel(pandasModel(strikes))
        pile_funcs.output_excel(self.strk_df, self.summ_df, self.run_df, xlsx_name)
        self.alr_proc = [f for f in glob.glob("*.wav")]

    def setOutputDir(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose directory")
        if dirname:
            self.outputDir = Path(str(dirname))
            self.populate_data(self.outputDir)
            self.message = f"Watching {str(dirname)}..."
            self.statusbar.showMessage(self.message)


class pandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)

        if index.column() > 0 and role == QtCore.Qt.TextAlignmentRole:
            return Qt.AlignVCenter + Qt.AlignRight
        elif index.column() == 0 and role == QtCore.Qt.TextAlignmentRole:
            return Qt.AlignVCenter + Qt.AlignRight

        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


class Emitter(QtCore.QObject):
    newDataFrameSignal = QtCore.pyqtSignal(pd.DataFrame)


class Watcher:
    def __init__(self, filename):
        # self.watch_dir = os.getcwd()
        # print(self.watch_dir)
        self.directory_to_watch = Path(filename)
        # print(self.directory_to_watch)
        self.emitter = Emitter()
        self.observer = Observer()
        self.event_handler = Handler(
            emitter=self.emitter,
            patterns=["*.wav"],
            ignore_patterns=["*.tmp"],
            ignore_directories=True,
        )

    def run(self):
        self.observer.schedule(
            self.event_handler, self.directory_to_watch, recursive=False
        )
        self.observer.start()


class Handler(PatternMatchingEventHandler):
    def __init__(self, *args, emitter=Emitter(), **kwargs):
        super(Handler, self).__init__(*args, **kwargs)
        self._emitter = emitter
        self.files = defaultdict(lambda: 0)

    def on_any_event(self, event):
        last_event = ""
        if event.is_directory:
            # print("This is the thing!")
            pass
        elif event.event_type == "created":
            # Take any action here when a file is first created.
            last_event = event.src_path
            df = pd.DataFrame()
            print("Received created event - %s." % event.src_path)
            self._emitter.newDataFrameSignal.emit(df.copy())
        elif event.event_type == "modified" and event.src_path.endswith((".wav")):
            if not event.src_path == last_event:  # files we haven't seen recently
                # do something
                print("api call with ", event.src_path)
                df = pd.DataFrame()
                self._emitter.newDataFrameSignal.emit(df.copy())
                last_event = event.src_path


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.setOutputDir()
    dir1 = ui.outputDir
    watcher = Watcher(ui.outputDir)
    watcher.run()
    watcher.emitter.newDataFrameSignal.connect(ui.repopulate_data)
    sys.exit(app.exec_())
