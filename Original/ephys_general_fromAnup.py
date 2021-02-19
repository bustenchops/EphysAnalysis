from ephys_class import EphysClass
import os
import re


datapath="/home/pi/ephys/"

expdate = "2021-02-16"

cellid = "C1"
# channels =  ['ImLEFT', 'IN1', 'IN7'] # olympus Rig
channels = ['ImRightP', 'VmRightS', 'IN6'] # LSM880
reschannel = [channels[1]]
clampchannel = [channels[0]]
print(os.listdir(datapath))
print(os.listdir(os.path.join(datapath,expdate)))
files = os.listdir(os.path.join(datapath,expdate))
abffiles = [f for f in files if re.search(".*.abf",f)]
abffiles.sort()
print(abffiles)
for abffile in abffiles:
    print("Opening %s"%abffile)
    ephys = EphysClass(os.path.join(datapath,expdate,abffile),loaddata=True)
    # EphysClass.seriesres_voltageclamp(ephys,'ImRightP',clampchannel)
    # ephys.info()
    ephys.show([0,2],[])           # [channels],[sweeps]
    # input()