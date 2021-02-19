import pyabf
import matplotlib.pyplot as plt

abf = pyabf.ABF("/home/pi/ephys/2021-02-16/2021_02_16_0009.abf")
# print(abf.headerText) # display header information in the console
# abf.headerLaunch() # display header information in a web browser


arf = 10
abf.setSweep(sweepNumber=arf, channel=0)
plt.figure(figsize=(8, 5))
plt.plot(abf.sweepX, abf.sweepY)
plt.show()