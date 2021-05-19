# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import pandas as pd
import os
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pile Monitor"))
        self.summLabel.setText(_translate("MainWindow", "Summary Strikes Report"))
        self.strkLabel.setText(_translate("MainWindow", "Peaks Report"))
        self.pushButton.setText(
            _translate("MainWindow", "Choose processing directory...")
        )

    def populate_data(self, dir1):
        _o, summary, strikes = pile_funcs.process_dir(dir1)
        strikes = strikes.reset_index().sort_values(axis=0, by=["filename", "ix"])
        self.summaryTable.setModel(pandasModel(summary))
        self.strikeTable.setModel(pandasModel(strikes))

    def repopulate_data(self):
        _o, summary, strikes = pile_funcs.process_dir(self.outputDir)
        strikes = strikes.reset_index().sort_values(axis=0, by=["filename", "ix"])
        self.summaryTable.setModel(pandasModel(summary))
        self.strikeTable.setModel(pandasModel(strikes))

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
        print(self.directory_to_watch)
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
    def __init__(self, *args, emitter=None, **kwargs):
        super(Handler, self).__init__(*args, **kwargs)
        self._emitter = emitter

    def on_any_event(self, event):
        if event.is_directory:
            print("This is the thing!")
        elif event.event_type == "created":
            # Take any action here when a file is first created.
            df = pile_funcs.process_dir(ui.outputDir)[2].reset_index()
            # print(df)
            print("Received created event - %s." % event.src_path)
            # df = pd.read_csv(event.src_path, header=1)
            self._emitter.newDataFrameSignal.emit(df.copy())
            # df.set_index(df.columns.values.tolist()[0], inplace=True)
            # append_df_to_excel(os.path.join(os.getcwd(), "myfile.xlsx"), df)
        elif event.event_type == "modified":
            # Taken any actionc here when a file is modified.
            df = pile_funcs.process_dir(ui.outputDir)[2].reset_index()
            # print("Received created event - %s." % event.src_path)
            # df = pd.read_csv(event.src_path, header=1)
            self._emitter.newDataFrameSignal.emit(df.copy())
            print("Received modified event - %s." % event.src_path)


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
