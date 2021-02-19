import pyabf
import re
import os
import matplotlib.pyplot as plt

def showoptions():
    selection = input('1. Show file info in browser: 2. Plot all sweeps: 3: Plot select sweeps')
    if selection == '1':
        showheader()
        showoptions()
    else:
        if selection == '2':
            plotall()
        else:
            if selection == '3':
                plotselected()
            else:
                print('invalid selection try again')
                showoptions()

def showheader():
    print(abf)

def plotall():
    i = 0
    j = 0
    k = 0
    tosweeps = input('How many sweeps in the file?')
    tosweepsint = int(tosweeps)
    fig = plt.figure(figsize=(10, 8))
    plt.title("All Sweeps in File")
    abf.setSweep(sweepNumber=1, channel=0)
    ax = fig.add_subplot(12, 1, (1, 4))
    ax.set_xlabel(abf.sweepLabelX)
    ax.set_ylabel(abf.sweepLabelY)
    abf.setSweep(sweepNumber=1, channel=1)
    bx = fig.add_subplot(12, 1, (6, 10))
    bx.set_xlabel(abf.sweepLabelX)
    bx.set_ylabel(abf.sweepLabelY)
    abf.setSweep(sweepNumber=1, channel=2)
    cx = fig.add_subplot(12, 1, 12)
    cx.set_xlabel(abf.sweepLabelX)
    cx.set_ylabel(abf.sweepLabelY)
    while i < tosweepsint:
        abf.setSweep(sweepNumber=i, channel=0)
        ax.plot(abf.sweepX, abf.sweepY, alpha=.5)
        i += 1
    while j < tosweepsint:
        abf.setSweep(sweepNumber=j, channel=1)
        bx.plot(abf.sweepX, abf.sweepY, alpha=.5)
        j += 1
    while k < tosweepsint:
        abf.setSweep(sweepNumber=k, channel=2)
        cx.plot(abf.sweepX, abf.sweepY, alpha=.5)
        k += 1
    plt.legend()
    plt.show()

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
if (os.path.exists(fileloc)):
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
foundfile = ''.join(abffiletoimport)
print(foundfile)
fullfilepath = os.path.join(fileloc, foundfile)
print('Full file path: ', fullfilepath)
abf = pyabf.ABF(fullfilepath)
showoptions()
