import pyabf
import matplotlib.pyplot as plt
import numpy as np

abf = pyabf.ABF("/home/pi/ephys/2021-02-16/2021_02_16_0009.abf")
# abf = pyabf.ABF("/home/2021_02_10_0003.abf")
# # print(abf.headerText) # display header information in the console
# # abf.headerLaunch() # display header information in a web browser
#
#
# arf = 0
# abf.setSweep(sweepNumber=arf, channel=1)
# plt.figure(figsize=(8, 5))
# plt.plot(abf.sweepX, abf.sweepY)
# plt.show()

# numsweep = int(input('How many sweeps to plot?'))
# sweeparray = np.zeros(numsweep, dtype=int)
# print(sweeparray)
# for x in range(numsweep):
#     sweeparray[x] = int(input('Sweep no:'))
# print(sweeparray)
print(abf.sweepPointCount)
