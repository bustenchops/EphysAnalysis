import pyabf
import re
import os
import numpy as np
from pathlib import Path
import pyqtgraph as pg

class EphysClass:
    def __init__(self):
        #loadfiles variables list (first initialized)
        self.homefolder = None          # input directory of ephys subfolders
        self.experdate = None           # input subfolder of ephys files to analyze
        self.fileloc = None             # joined directory homefolder + experdate
        self.fileloclist = None         # list of files in fileloc directory
        self.abffiles = None            # list of abf files in the array fileloclist
        self.abffiletoimport = None     # input the 4 digit file number to select file
        self.foundfile = None           # sting of the abf file name located from the search
        self.fullfilepath = None        # joined full path to the desired file fileloc + foundfile
        self.abf = None                 #instance of pyabf.ABF
        #showoptions variable list (first initialized)
        self.selection = None           # input selection for analysis
        self.dict = None                # dictionary map for selection of analysis
        #neworrepeat variable list (first initialized)
        self.choices = None             # input selection for repeating or new file
        self.dictch = None              # dictionary map for selection
        #showheader variable list (first initialized)
        #showheaderfull variable list (first initialized)
        self.numchan = None             # used to store number of channels
        #plotall variable list (first initialized)
        self.fig = None                 # intance of plt.figure
        self.ax = None                  # fig subplot 1
        self.bx = None                  # fig subplot 2
        self.cx = None                  # fig subplot 3
        #plotselected variable list (first intialized)
        self.numsweep = None            # input indicate number of sweeps to analyze
        self.sweeparray = None          # initialize array with size indicated by numsweep
        #threedee variable list (first initialized)
        self.i1 = None                  # initialize the plot array for 3d
        self.i2 = None                  # initialize the plot array for 3d calc the
        self.dataX = None               # used to set offset for the 3d plots
        self.dataY = None               # used to set offset for the 3d plots
        #sumandsave variable list (first initialized)
        self.filename = None            # used to convert filename so it can be use to save the image
        self.filename_replace = None    # used to replace abf extension with jpg for fig saving

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Ephys - Quick Analysis")

        label = QLabel("Hello!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)

        button_action = QAction("Your button", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(button_action)

        self.dir_path = QFileDialog.getExistingDirectory() ## use this to select folder.




