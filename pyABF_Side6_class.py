import pyabf
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#This has all the working bits

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


    def loadfiles(self):
        print('IMPORTING DIRECTORIES')
        self.homefolder = input('Are the files in the folder /home/pi/ephys/ (y or n)? ')
        if self.homefolder == 'y':
            self.homefolder = '/home/pi/ephys/'
        else:
            self.homefolder = input('enter the datapath in /folder/subfolder/ format:')
            print('Experiments should be stored in subfolder by date in yyyy-mm-dd format.')
        self.experdate = input('Enter the experiment date in yyyy-mm-dd format ')
        print('Subfolders within home folder:')
        print(os.listdir(self.homefolder))
        self.fileloc = os.path.join(self.homefolder, self.experdate)
        if os.path.exists(self.fileloc):
            print('Files within experiment date folder')
            print(os.listdir(os.path.join(self.homefolder, self.experdate)))
            self.fileloclist = os.listdir(os.path.join(self.homefolder, self.experdate))
        else:
            print('The path does not exist, check the home path and date.')
        print('ISOLATING ABF FILES...')
        self.abffiles = [extension for extension in self.fileloclist if re.search("\.abf$", extension)]
        self.abffiles.sort()
        print('*.abf files in folder:')
        print(self.abffiles)
        self.abffiletoimport = input('Type in file number (ex:0012)')
        print('Searching files:')
        self.abffiletoimport = [file for file in self.abffiles if re.search(self.abffiletoimport, file)]
        print('Found: ', self.abffiletoimport)
        # converts list to string
        self.foundfile = ''.join(self.abffiletoimport)
        print(self.foundfile)
        self.fullfilepath = os.path.join(self.fileloc, self.foundfile)
        print('Full file path: ', self.fullfilepath)
        self.abf = pyabf.ABF(self.fullfilepath)
        self.showoptions()

    def showoptions(self):
        print('1. Show file info')
        print('2. Show header in web browser')
        print('3. Plot all sweeps')
        print('4. Plot select sweeps')
        print('5. Plot all sweeps in 3D')
        print('6. Take all files, plot in 3d and save')
        self.selection = int(input('Enter selection: '))
        if self.selection <= 6:
            self.dict = {
                1: self.showheader,
                2: self.showheaderfull,
                3: self.plotall,
                4: self.plotselected,
                5: self.threedee,
                6: self.sumandsave,
            }
            self.dict.get(self.selection)()
        else:
            print('invalid selection try again')
            self.neworpeat()

    def neworpeat(self):
        self.choices = int(input('1: rehash same file 2: new file'))
        if self.choices <= 6:
            self.dictch = {
                1: self.showoptions,
                2: self.loadfiles,
            }
            self.dictch.get(self.choices)()
        else:
            print('invalid selection try again')
            self.neworpeat()

    def showheader(self):
        print()
        print('File info START:')
        print(self.abf)
        print(':File info END')
        print()
        self.neworpeat()

    def showheaderfull(self):
        self.numchan = self.abf.channelCount
        print('Opening full header for each channel present')
        for thechan in range(self.numchan):
            self.abf.setSweep(sweepNumber=0, channel=thechan)
            self.abf.headerLaunch()
        self.neworpeat()

    def plotall(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle('All Sweeps in File')
        self.abf.setSweep(sweepNumber=0, channel=0)
        self.ax = self.fig.add_subplot(12, 1, (1, 8))
        # ax.set_xlabel(abf.sweepLabelX)
        self.ax.set_ylabel(self.abf.sweepLabelY)
        self.abf.setSweep(sweepNumber=0, channel=1)
        self.bx = self.fig.add_subplot(12, 1, (9, 10))
        # bx.set_xlabel(abf.sweepLabelX)
        self.bx.set_ylabel(self.abf.sweepLabelY)
        self.abf.setSweep(sweepNumber=0, channel=2)
        self.cx = self.fig.add_subplot(12, 1, (11, 12))
        self.cx.set_xlabel(self.abf.sweepLabelX)
        self.cx.set_ylabel(self.abf.sweepLabelY)
        for sweepnum in self.abf.sweepList:
            self.abf.setSweep(sweepnum, channel=0)
            self.ax.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5)
        self.abf.setSweep(0, channel=1)
        self.bx.plot(self.abf.sweepX, self.abf.sweepC, alpha=.5)
        self.abf.setSweep(0, channel=2)
        self.cx.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5)
        plt.show()
        self.neworpeat()

    def plotselected(self):
        self.numsweep = int(input('How many sweeps to plot?'))
        self.sweeparray = np.zeros(self.numsweep, dtype=int)
        for x in range(self.numsweep):
            self.sweeparray[x] = int(input('Sweep no:'))
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle('Selected Sweeps from File')
        self.abf.setSweep(sweepNumber=1, channel=0)
        self.ax = self.fig.add_subplot(12, 1, (1, 8))
        # ax.set_xlabel(abf.sweepLabelX)
        self.ax.set_ylabel(self.abf.sweepLabelY)
        self.abf.setSweep(sweepNumber=1, channel=1)
        self.bx = self.fig.add_subplot(12, 1, (9, 10))
        # bx.set_xlabel(abf.sweepLabelX)
        self.bx.set_ylabel(self.abf.sweepLabelY)
        self.abf.setSweep(sweepNumber=1, channel=2)
        self.cx = self.fig.add_subplot(12, 1, (11, 12))
        self.cx.set_xlabel(self.abf.sweepLabelX)
        self.cx.set_ylabel(self.abf.sweepLabelY)
        for sweepnum in self.sweeparray:
            self.abf.setSweep(sweepnum, channel=0)
            self.ax.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5, label="Sweep %d" % sweepnum)
        self.abf.setSweep(0, channel=1)
        self.bx.plot(self.abf.sweepX, self.abf.sweepC, alpha=.5)
        self.abf.setSweep(0, channel=2)
        self.cx.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5)
        self.ax.legend()
        plt.show()
        self.neworpeat()

    def threedee(self):
        self.fig = plt.figure(figsize=(10, 5))
        self.fig.suptitle('All Sweeps in File')
        self.abf.setSweep(sweepNumber=0, channel=0)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis('Off')
        for sweepnum in self.abf.sweepList:
            self.abf.setSweep(sweepnum, channel=0)
            self.i1, self.i2 = 0, int(self.abf.dataRate * self.abf.sweepLengthSec)
            self.dataX = self.abf.sweepX[self.i1:self.i2] + .025 * sweepnum
            self.dataY = self.abf.sweepY[self.i1:self.i2] + 10 * sweepnum
            self.ax.plot(self.dataX, self.dataY, alpha=.5)
        plt.show()
        self.neworpeat()

    def sumandsave(self):
        for abflist in self.abffiles:
            self.foundfile = ''.join(abflist)
            print(self.foundfile)
            self.fullfilepath = os.path.join(self.fileloc, self.foundfile)
            self.abf = pyabf.ABF(self.fullfilepath)
            self.filename = Path(self.fullfilepath)
            self.filename_replace = self.filename.with_suffix('.jpg')
            print(self.filename)
            self.fig = plt.figure(figsize=(10, 5))
            self.fig.suptitle(abflist)
            self.abf.setSweep(sweepNumber=0, channel=0)
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.axis('Off')
            for sweepnum in self.abf.sweepList:
                self.abf.setSweep(sweepnum, channel=0)
                self.i1, self.i2 = 0, int(self.abf.dataRate * self.abf.sweepLengthSec)
                self.dataX = self.abf.sweepX[self.i1:self.i2] + .025 * sweepnum
                self.dataY = self.abf.sweepY[self.i1:self.i2] + 15 * sweepnum
                self.ax.plot(self.dataX, self.dataY, alpha=.5)
            self.fig.savefig(self.filename_replace)
            plt.close()
            self.neworpeat()