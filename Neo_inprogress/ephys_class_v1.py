from neo import AxonIO
import numpy as np
import re
import os

class filemanage:

    def __init__(self):
       self.fileloc = None

    def directimport(self):
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
        if (os.path.exists(self.fileloc)):
            print('Files within experiment date folder')
            print(os.listdir(os.path.join(self.homefolder, self.experdate)))
            self.fileloclist = os.listdir(os.path.join(self.homefolder, self.experdate))
        else:
            print('The file does not exist, check the path and date.')
            return ()

    def findabffile(self):
        print('ISOLATING ABF FILES...')
        self.abffiles = [extension for extension in self.fileloclist if re.search("\.abf$", extension)]
        self.abffiles.sort()
        print('*.abf files in folder:')
        print(self.abffiles)

    def importabf(self):
        self.abffiletoimport = input('Type (a) to process all files in folder, otherwise type in file number (ex:0012)')
        if self.abffiletoimport != 'a':
            print ('Searching files:')
            self.abffiletoimport = [file for file in self.abffiles if re.search(self.abffiletoimport, file)]
            print('Found: ', self.abffiletoimport)
            self.foundfile = ''.join(self.abffiletoimport)
            print(self.foundfile)
            self.fullfilepath = os.path.join(self.fileloc, self.foundfile)
            print('Full file path: ', self.fullfilepath)
            self.abf = AxonIO(filename=self.fullfilepath)
        else:
            pass

    def exploreabf(self):
        self.numblocks = self.abf.block_count()
        self.samplingrate = self.abf.get_signal_sampling_rate()
        self.abf.
