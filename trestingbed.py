from neo import AxonIO
import numpy as np
import re
import os

homefolder = '/home/pi/ephys/2021-02-16/2021_02_12_0009.abf'
abf = AxonIO(filename=homefolder)