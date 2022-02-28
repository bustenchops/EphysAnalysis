import sys
from PySide6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QToolBar, QStatusBar,
    QFileDialog, QComboBox, QGridLayout, QWidget
)
from PySide6.QtGui import (
    QAction
)
from PySide6.QtCore import Qt
import pyabf
import re
import os
import numpy as np
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ## declared variables
        self.dir_path = "None selected"
        self.filelist = ["Select Folder"]
        self.filelistindex = None

        self.setWindowTitle("Ephys-Quick Analysis")

        layout = QGridLayout()

        toolbar = QToolBar("Function toolbar")
        self.addToolBar(toolbar)
        self.folderbuttonsetup()
        toolbar.addAction(self.folderbutton)
        self.setStatusBar(QStatusBar(self))

        self.filebox = QComboBox()
        self.filebox.addItems(self.filelist)
        layout.addWidget(self.filebox, 0, 0, 1, 3)
        self.filebox.currentTextChanged.connect(self.fileselected)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def folderbuttonsetup(self):
        self.folderbutton = QAction("Folder", self)
        self.folderbutton.setStatusTip(self.dir_path)
        self.folderbutton.triggered.connect(self.folderselect)

    def folderselect(self):
        print("click")
        print(self.dir_path)
        self.dir_path = QFileDialog.getExistingDirectory()
        self.folderbutton.setStatusTip(self.dir_path)
        print(self.dir_path)
        self.fileboxlist()

    def fileboxlist(self):
        print('Files within experiment date folder')
        print('ISOLATING ABF FILES...')
        self.dir_path_1 = self.dir_path[2:] + "/"
        self.fileinfolder = os.listdir(self.dir_path_1)
        self.abffiles = [extension for extension in self.fileinfolder if re.search("\.abf$", extension)]
        self.abffiles.sort()
        print('*.abf files in folder:')
        print(self.abffiles)
        self.filebox.clear()

        self.filelist = self.abffiles
        self.filebox.addItems(self.filelist)

    def fileselected(self, i):
        print(i)








app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec_()
