from neo import AxonIO
import numpy as np
import re
import os

homefolder = '/home/pi/ephys/2021-02-16/2021_02_16_0009.abf'
# abf = AxonIO(filename=homefolder)
reader = AxonIO(filename=homefolder)
nblocks = reader.block_count()
si = reader.get_signal_sampling_rate()
samplesize = reader.get_signal_size(block_index=0, seg_index=0)
tstart = reader.segment_t_start(block_index=0, seg_index=0)
tstop = reader.segment_t_stop(block_index=0, seg_index=0)
header = reader.header
nchannels = reader.signal_channels_count()

gseg = reader.segment_count(block_index=0)
channelspecs = reader.header["signal_channels"].dtype.names

# bl = reader.read_block(lazy= False)
seg = reader.read_segment(block_index=0, seg_index=1)
print(seg.analogsignals)



protocol = reader.read_protocol()
rawprotocol = reader.read_raw_protocol()
print('numberblocks: ', nblocks)
print('samplerate: ', si)
print('timestart: ', tstart)
print('timestop: ', tstop)
print('haeder: ', header)
print('channels: ', nchannels)
print('protocol: ', protocol)
print('rawprotocol: ', protocol)
print('numsegments: ', gseg)
print('channelspecs: ', channelspecs)