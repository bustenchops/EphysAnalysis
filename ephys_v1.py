from neo import AxonIO
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
import re
from scipy import interpolate
from scipy.signal import butter, lfilter, freqz, detrend
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import plotting_functions
import os