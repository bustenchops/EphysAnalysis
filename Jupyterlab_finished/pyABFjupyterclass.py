import pyabf
import re
import os
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

#This has all the working bits

class EphysClass:
    def __init__(self):
        #loadfiles variables list (first initialized)
        self.homefolder = None          # input directory of ephys subfolders
        self.experdate = None           # input subfolder of ephys files to analyze
        self.fileloc = None             # joined directory homefolder + experdate
        self.fileloclist = None         # list of files in fileloc directory
        # self.abffiles = None            # list of abf files in the array fileloclist
        self.currentfile = None         #Current file name plotted
        self.fullfilepath = None  # joined full path to the desired file fileloc + foundfile
        self.abf = None  # instance of pyabf.ABF
        self.currentchannel = 0         #default channel for first file
        self.currentsweep = None           #default sweep value for first file
        self.numsweep = None               # number of sweeps
        self.abffiletoimport = None     # input the 4 digit file number to select file
        self.foundfile = None           # sting of the abf file name located from the search
        #plotall variable list (first initialized)
        self.fig = None                 # intance of plt.figure
        self.ax = None                  # fig subplot 1
        self.bx = None                  # fig subplot 2
        self.cx = None                  # fig subplot 3
        #plotselected variable list (first intialized)
        self.numsweep = None            # input indicate number of sweeps to analyze



    def loadfiles(self):
        print('IMPORTING DIRECTORIES....')
        self.homefolder = input('Are the files in the folder /home/pi/ephys/ (y or n)? ')
        if self.homefolder == 'y':
            self.homefolder = '/home/pi/ephys/'
        else:
            self.homefolder = input('enter the datapath in /folder/subfolder/ format:')
            print('Experiments should be stored in subfolder by date in yyyy-mm-dd format.')
        self.experdate = input('Enter the experiment date in yyyy-mm-dd format ')
        # print('Subfolders within home folder:')
        # print(os.listdir(self.homefolder))
        self.fileloc = os.path.join(self.homefolder, self.experdate)
        # print('MERGING INPUT FOR HOME AND EXPERIMENT FOLDER NAMES...')
        if os.path.exists(self.fileloc):
            # print('Files within experiment date folder')
            # print(os.listdir(os.path.join(self.homefolder, self.experdate)))
            self.fileloclist = os.listdir(os.path.join(self.homefolder, self.experdate))
        else:
            print('The path does not exist, check the home path and date.')
        # print('ISOLATING ABF FILES...')
        self.abffiles = [extension for extension in self.fileloclist if re.search("\.abf$", extension)]
        self.abffiles.sort()
        # print('*.abf files in folder:')
        # print(self.abffiles)
        # self.abffiletoimport = self.abffiles[0]
        # print('Default file name:')
        # print(self.abffiles[0])
        # self.abffiletoimport = self.abffiles[0]
        # print(abffiletoimport)

        self.ddfile = interact(self.thefile, dropdownfile=widgets.Dropdown(options=self.abffiles, values=1, description='File:'),)

    def thefile(self, dropdownfile):
        self.fullfilepath = os.path.join(self.fileloc, dropdownfile)
        self.currentfile = dropdownfile
        self.abf = pyabf.ABF(self.fullfilepath)
        self.numsweep = self.abf.sweepCount - 1
        self.slidsweep = interact(self.setsweep, sweeppick=widgets.IntSlider(
            min=0,
            max=self.numsweep,
            values=1,
            step=1,
            description='Sweep:',),)
        # self.jupyterplot()

    # def setchannel(self, channelpick):
    #     self.currentchannel = channelpick

    def setsweep(self, sweeppick):
        self.currentsweep = sweeppick
        self.jupyterplot()


    def jupyterplot(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle(self.currentfile)
        self.ax = self.fig.add_subplot(3, 1, (1, 2))
        self.ax.set_xlabel(self.abf.sweepLabelX)
        self.ax.set_ylabel(self.abf.sweepLabelY)
        for sweepnum in self.abf.sweepList:
            self.abf.setSweep(sweepnum, channel=0)
            if sweepnum == self.currentsweep:
                self.ax.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5, color='r')
            else:
                self.ax.plot(self.abf.sweepX, self.abf.sweepY, alpha=.5, color='c')
        self.bx = self.fig.add_subplot(3, 1, (3))
        self.bx.set_ylabel(self.abf.sweepLabelY)
        for sweepnum in self.abf.sweepList:
            self.abf.setSweep(sweepnum, channel=1)
            if sweepnum == self.currentsweep:
                self.bx.plot(self.abf.sweepX, self.abf.sweepC, alpha=.5, color='r')
            else:
                self.bx.plot(self.abf.sweepX, self.abf.sweepC, alpha=.5, color='c')
        # self.fig.canvas.draw()

