import pyabf
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class EphysClass:
    def __init__(self):
        #loadfiles variables (first initialized)
        self.homefolder = None          # input directory of ephys subfolders
        self.experdate = None           # input subfolder of ephys files to analyze
        self.fileloc = None             # joined directory homefolder + experdate
        self.fileloclist = None         # list of files in fileloc directory
        self.abffiles = None            # list of abf files in the array fileloclist
        self.abffiletoimport = None     # input the 4 digit file number to select file
        self.foundfile = None           # sting of the abf file name located from the search
        self.fullfilepath = None        # joined full path to the desired file fileloc + foundfile
        #showoptions variable (first initialized)
        self.selection = None           # input selection for analysis
    def loadfiles(self):
        print('IMPORTING DIRECTORIES')
        self.homefolder = input('Are the files in the folder /home/pi/ephys/ (y or n)?')
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
        showoptions()

    def showoptions(self):
        self.selection = input('1. Show file info: 2. Plot all sweeps: 3. Plot select sweeps: 4. Plot all sweeps in 3D: 5. Take all in 3d and save')
        if self.selection == '1':
            showheader()
            showoptions()
        elif selection == '2':
               plotall()
        elif selection == '3':
             plotselected()
        elif selection == '4':
            threedee()
        elif selection == '5':
            sumandsave()
        else:
            print('invalid selection try again')
            showoptions()


    def showheader():
        print(abf)


    def plotall():
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('All Sweeps in File')
        abf.setSweep(sweepNumber=1, channel=0)
        ax = fig.add_subplot(12, 1, (1, 8))
        # ax.set_xlabel(abf.sweepLabelX)
        ax.set_ylabel(abf.sweepLabelY)
        abf.setSweep(sweepNumber=1, channel=1)
        bx = fig.add_subplot(12, 1, (9, 10))
        # bx.set_xlabel(abf.sweepLabelX)
        bx.set_ylabel(abf.sweepLabelY)
        abf.setSweep(sweepNumber=1, channel=2)
        cx = fig.add_subplot(12, 1, (11, 12))
        cx.set_xlabel(abf.sweepLabelX)
        cx.set_ylabel(abf.sweepLabelY)
        for sweepnum in abf.sweepList:
            abf.setSweep(sweepnum, channel=0)
            ax.plot(abf.sweepX, abf.sweepY, alpha=.5)
        abf.setSweep(0, channel=1)
        bx.plot(abf.sweepX, abf.sweepC, alpha=.5)
        abf.setSweep(0, channel=2)
        cx.plot(abf.sweepX, abf.sweepY, alpha=.5)
        plt.show()


    def plotselected():
        numsweep = int(input('How many sweeps to plot?'))
        sweeparray = np.zeros(numsweep, dtype=int)
        for x in range(numsweep):
            sweeparray[x] = int(input('Sweep no:'))
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('Selected Sweeps from File')
        abf.setSweep(sweepNumber=1, channel=0)
        ax = fig.add_subplot(12, 1, (1, 8))
        # ax.set_xlabel(abf.sweepLabelX)
        ax.set_ylabel(abf.sweepLabelY)
        abf.setSweep(sweepNumber=1, channel=1)
        bx = fig.add_subplot(12, 1, (9, 10))
        # bx.set_xlabel(abf.sweepLabelX)
        bx.set_ylabel(abf.sweepLabelY)
        abf.setSweep(sweepNumber=1, channel=2)
        cx = fig.add_subplot(12, 1, (11, 12))
        cx.set_xlabel(abf.sweepLabelX)
        cx.set_ylabel(abf.sweepLabelY)
        for sweepnum in sweeparray:
            abf.setSweep(sweepnum, channel=0)
            ax.plot(abf.sweepX, abf.sweepY, alpha=.5, label="Sweep %d" % sweepnum)
        abf.setSweep(0, channel=1)
        bx.plot(abf.sweepX, abf.sweepC, alpha=.5)
        abf.setSweep(0, channel=2)
        cx.plot(abf.sweepX, abf.sweepY, alpha=.5)
        ax.legend()
        plt.show()


    def threedee():
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('All Sweeps in File')
        abf.setSweep(sweepNumber=1, channel=0)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('Off')
        for sweepnum in abf.sweepList:
            abf.setSweep(sweepnum, channel=0)
            i1, i2 = 0, int(abf.dataRate * 1.5)
            dataX = abf.sweepX[i1:i2] + .025 * sweepnum
            dataY = abf.sweepY[i1:i2] + 10 * sweepnum
            ax.plot(dataX, dataY, alpha=.5)
        plt.show()


    def sumandsave():
        for abflist in abffiles:
            foundfile = ''.join(abflist)
            print(foundfile)
            fullfilepath = os.path.join(fileloc, foundfile)
            abf = pyabf.ABF(fullfilepath)
            filename = Path(fullfilepath)
            filename_replace = filename.with_suffix('.jpg')
            print(filename)
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle(abflist)
            abf.setSweep(sweepNumber=1, channel=0)
            ax = fig.add_subplot(1, 1, 1)
            ax.axis('Off')
            for sweepnum in abf.sweepList:
                abf.setSweep(sweepnum, channel=0)
                i1, i2 = 0, int(abf.dataRate * 1.5)
                dataX = abf.sweepX[i1:i2] + .025 * sweepnum
                dataY = abf.sweepY[i1:i2] + 10 * sweepnum
                ax.plot(dataX, dataY, alpha=.5)
            fig.savefig(filename_replace)


    print('IMPORTING DIRECTORIES')
    homefolder = input('Are the files in the folder /home/pi/ephys/ (y or n)?')
    if homefolder == 'y':
        homefolder = '/home/pi/ephys/'
    else:
        homefolder = input('enter the datapath in /folder/subfolder/ format:')
        print('Experiments should be stored in subfolder by date in yyyy-mm-dd format.')
    experdate = input('Enter the experiment date in yyyy-mm-dd format ')
    print('Subfolders within home folder:')
    print(os.listdir(homefolder))
    fileloc = os.path.join(homefolder, experdate)
    if os.path.exists(fileloc):
        print('Files within experiment date folder')
        print(os.listdir(os.path.join(homefolder, experdate)))
        fileloclist = os.listdir(os.path.join(homefolder, experdate))
    else:
        print('The file does not exist, check the path and date.')


