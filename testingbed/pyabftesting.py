import pyabf
import matplotlib.pyplot as plt
import numpy as np
import re

# abf = pyabf.ABF("/home/pi/ephys/2021-02-16/2021_02_16_0009.abf")
abf.setSweep(sweepNumber=1, channel=1)
abf = pyabf.ABF("/home/2021_02_10_0003.abf")
# print(abf.headerText) # display header information in the console
abf.headerLaunch() # display header information in a web browser
#         si = abf.headerText
print(abf.sweepLengthSec)
#         print(si)
#         finditall = re.findall(r"dataSecPerPoint = \S+", si)
#         sss = finditall[0]
#         print(sss)
#         # print(finditall)
#         split = re.split(r'\s',sss)
#         sampleinterval = split[2]
#         npversion = float(sampleinterval)
#
#         print(npversion)
#         kirk = npversion * 10
#         print(kirk)
#
#
# xvariablesweep = np.array(abf.sweepX)
# yvariablesweep = np.array(abf.sweepY)
# arf = 2
# abf.setSweep(sweepNumber=arf, channel=1)
# plt.figure(figsize=(8, 5))
# plt.plot(xvariablesweep, yvariablesweep)
# # plt.plot(abf.sweepX, abf.sweepD(2))
# plt.show()

# numsweep = int(input('How many sweeps to plot?'))
# sweeparray = np.zeros(numsweep, dtype=int)
# print(sweeparray)
# for x in range(numsweep):
#     sweeparray[x] = int(input('Sweep no:'))
# print(sweeparray)
# print(abf.sweepPointCount)
#
# print (dir(pyabf.ABF))



