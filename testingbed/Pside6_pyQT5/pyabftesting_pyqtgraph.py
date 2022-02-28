import pyabf
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import PySide6
import pyqtgraph as pg



def showoptions():
    selection = int(input('1. Show file info: 2. Plot all sweeps: 3. Plot select sweeps: 4. Plot all sweeps in 3D: 5. Take all in 3d and save'))
    # if selection == '1':
    #     showheader()
    #     showoptions()
    # else:
    #     if selection == '2':
    #         plotall()
    #     else:
    #         if selection == '3':
    #             plotselected()
    #         else:
    #             if selection == '4':
    #                 threedee()
    #             else:
    #                 if selection == '5':
    #                     sumandsave()
    #                 else:
    #                     print('invalid selection try again')
    #                     showoptions()
    if selection <= 6:
        dict = {
            1 : showheader,
            2 : pyqttest,
            3 : plotselected,
            4 : threedee,
            5 : sumandsave,
            6 : pyqttest,
            }
        dict.get(selection)()
    else:
        print('invalid selection try again')
        showoptions()

def showheader():
    print()
    print('File info START:')
    print(abf)
    print(':File info END')
    print()
    showoptions()


def pyqttest():
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

print('ISOLATING ABF FILES...')
abffiles = [extension for extension in fileloclist if re.search("\.abf$", extension)]
abffiles.sort()
print('*.abf files in folder:')
print(abffiles)
abffiletoimport = input('Type in file number (ex:0012)')
print('Searching files:')
abffiletoimport = [file for file in abffiles if re.search(abffiletoimport, file)]
print('Found: ', abffiletoimport)
# converts list to string
foundfile = ''.join(abffiletoimport)
print(foundfile)
fullfilepath = os.path.join(fileloc, foundfile)
print('Full file path: ', fullfilepath)
abf = pyabf.ABF(fullfilepath)
showoptions()
