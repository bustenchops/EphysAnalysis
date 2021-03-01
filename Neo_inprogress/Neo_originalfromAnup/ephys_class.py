from neo import AxonIO
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import re
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from Neo_inprogress.Neo_originalfromAnup import plotting_functions
import os


def smooth(y, windowlen=3, window='hanning'):
    s = np.concatenate([y[windowlen - 1:0:-1], y, y[-2:-windowlen - 1:-1]])
    if (window == 'flat'):
        w = np.ones(windowlen, dtype='double')
    else:
        w = eval('np.' + window + '(windowlen)')

    yy = np.convolve(w / w.sum(), s, mode='same')[windowlen - 1:-windowlen + 1]
    return (yy)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def template_match(t, y, tt, yy):
    # fit a template to a given trace. returns icorrmac and beta
    si = np.diff(t, axis=0).mean()
    isearch = len(y) - len(yy)
    corr = np.zeros(isearch)
    # y = y - y.max()
    # yy = yy - yy.max()
    # normalize response so that the correlation is amplitude independent
    yn = (y - y.mean(axis=0)) / y.std(axis=0)
    yyn = (yy - yy.mean(axis=0)) / yy.std(axis=0)
    # yn = yn - yn[0,0]
    # yyn = yyn - yyn[0,0]
    # yn = yn - yn.max()
    # yyn = yyn - yyn.max()
    # perform correlation between the actual trace and the template
    for k in np.arange(0, isearch):
        corr[k] = np.correlate(yn[k:(k + len(yy)), 0], yyn[:, 0])
        # get the index at which the correlation is maximum
    icorrmin = corr.argmin()
    if (icorrmin > 0):
        icorrmax = corr[0:icorrmin].argmax()
    else:
        icorrmax = corr.argmax()
        print(len(y), len(yy), isearch, icorrmax)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(corr)
        # plt.show()
        # solve normal equation/lstsqr to estimate beta
    beta = np.linalg.lstsq(yy, y[icorrmax:icorrmax + len(yy)], rcond=None)[0][0]
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ah.plot(t,y)
    # tt = np.arange(0,len(yy)*si,si)
    # ah.plot(tt+t[icorrmax],yy*beta)
    # plt.show()
    print(icorrmax, beta[0])
    return (icorrmax, beta[0], yy)


class EphysClass:

    def __init__(self, fname, loaddata=True, badsweeps=-1, vclamp=None, cclamp=None, fid=None):
        self.fname = fname
        self.fid = fid
        if (fid == None):
            self.fid = re.search('[0-9]{1,10}.abf', fname)[0][0:-4]
            # ------------
        self.loaddata = loaddata
        self.badsweeps = badsweeps
        self.vclamp = vclamp
        self.cclamp = cclamp
        if (not os.path.exists(self.fname)):
            print("The file does not exist! check the path!")
            return ()
        # -----------
        self.reader = AxonIO(filename=self.fname)
        self.nblocks = self.reader.block_count()
        self.nsweeps = np.zeros(self.nblocks, dtype=np.int)
        self.si = 1 / self.reader.get_signal_sampling_rate()
        self.samplesize = self.reader.get_signal_size(block_index=0, seg_index=0)
        self.tstart = self.reader.segment_t_start(block_index=0, seg_index=0)
        self.tstop = self.reader.segment_t_stop(block_index=0, seg_index=0)
        self.header = self.reader.header
        # get number of sweeps in each block
        for i in np.arange(0, self.nblocks):
            self.nsweeps[i] = self.reader.segment_count(i)
        if (len(self.nsweeps) == 1):
            self.nsweeps = self.nsweeps[0]
            self.nchannels = self.reader.signal_channels_count()
            self.sres = np.zeros((int(self.nsweeps)))
            self.nstim = 0
            self.stimprop = [dict({"nstim": 0, "isi": 0, "istims": [], "tstims": []}) for sweep in
                             np.arange(0, self.nsweeps)]
            self.clampprop = [dict({"iclamps": [], "tclamps": []}) for sweep in np.arange(0, self.nsweeps)]
            # initialize good sweeps
        self.goodsweeps = np.setdiff1d(np.arange(0, self.nsweeps), self.badsweeps)
        # extract channel/signal properties for each signal/channel from header
        channelspecs = self.reader.header["signal_channels"].dtype.names
        # print(channelspecs)
        # print(self.nchannels)
        self.channelspecs = [dict.fromkeys(channelspecs) for channel in np.arange(0, self.nchannels)]
        for i in np.arange(0, self.nchannels):
            self.channelspecs[i] = self.reader.header["signal_channels"][i]
            # ----
            # get number of samples from the first block, first segment
        self.nsamples = self.reader.get_signal_size(block_index=0, seg_index=0)
        # print(self.channelspecs)
        # input()
        if (self.loaddata):
            self.data = self.__loaddata()
            # extract date/time of the file
        self.datetime = self.reader.raw_annotations["blocks"][0]["rec_datetime"]
        self.date = self.datetime.date().isoformat()
        self.time = self.datetime.time().isoformat()
        # ============================================

    def average_trace(self, reschan, trgchan, tpre, tpost, success_sweeps=[]):
        if (len(self.sweeps) == 0):
            return ()
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschan, channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchan, channelspec['name'])]
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.sweeps])
        iisi = np.array([int(self.stimprop[sweep]["isi"] / self.si) for sweep in self.sweeps])
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.sweeps])
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.sweeps])
        ipost = int(tpost / self.si)
        ipre = int(tpre / self.si)
        sweeps = self.sweeps
        yy = np.zeros((ipost - ipre, len(sweeps) * tstims.shape[0]))
        itrace = 0
        print("success_sweeps: ", success_sweeps)
        for i in np.arange(0, len(sweeps)):
            y = self.data[sweeps[i], iresch, :].T
            y.resize(len(y))
            t = np.arange(0, y.shape[0] * self.si, self.si)[:, np.newaxis]
            # get baseline by linear fitting
            y = smooth(y, windowlen=21, window='hanning')
            m1, c1 = np.linalg.lstsq(np.concatenate((t, np.ones((len(t), 1))), axis=1), y, rcond=None)[0]
            baseline = (t * m1 + c1)
            y = y - baseline[:, 0]
            for j in np.arange(0, len(tstims[i])):
                if (j == (len(tstims[i]) - 1)):
                    y2 = y[istims[i][j] + ipre:(istims[i][j] + ipost)]
                    t2 = t[istims[i][j] + ipre:(istims[i][j] + ipost)]
                else:
                    y2 = y[istims[i][j] + ipre:istims[i][j + 1]]
                    t2 = t[istims[i][j] + ipre:istims[i][j + 1]]
                    # detrend(y,axis=0,type='linear',overwrite_data=True)
                yy[:, itrace] = y2
                itrace = itrace + 1
        if (len(success_sweeps) > 0):
            yyavg = yy[:, success_sweeps].mean(axis=1)[:, np.newaxis]
        else:
            yyavg = yy.mean(axis=1)[:, np.newaxis]
            ttavg = np.arange(0, (len(yyavg)) * self.si, self.si)[:, np.newaxis]
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(ttavg,yyavg,color='k')
            # ah.set_title(str(int(isi.mean()*1000))+" ms")
            # plt.show()
        return (yyavg)

    def find_peaks(self, reschan, trgchan, plotdata=True, peakdir='+'):
        if (len(self.sweeps) == 0):
            return ([], [], [])
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschan, channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchan, channelspec['name'])]
        # -------------
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.sweeps])
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.sweeps])
        tpre = 0.01
        tpost = 0.1
        ipost = int(tpost / self.si)
        tstimsfst = tstims - tpre
        ipre = int(tpre / self.si)
        istimsfst = istims - ipre
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.sweeps])
        iisi = np.array([int(self.stimprop[sweep]["isi"] / self.si) for sweep in self.sweeps if
                         not np.isnan(self.stimprop[sweep]["isi"])])
        # ------------
        if (all(isi) > 0):
            tstimslst = [min(tstims[i][j] + tpost, tstims[i][j] + isi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(tstims[i]))]
        else:
            tstimslst = [max(tstims[i][j] + tpost, tstims[i][j] + isi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(tstims[i]))]
            tstimslst = np.array(tstimslst)
            tstimslst = tstimslst.reshape(tstimsfst.shape)
            tpost = [min(tstims[i][-1] + tpost, tstims[i][-1] + isi[i]) for i in np.arange(0, len(self.sweeps))]
            tpost = np.array(tpost)[:, np.newaxis]
            # -------------
        if (all(isi) > 0):
            istimslst = [min(istims[i][j] + ipost, istims[i][j] + iisi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(istims[i]))]
        else:
            istimslst = [max(istims[i][j] + ipost, istims[i][j] + iisi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(istims[i]))]
            istimslst = np.array(istimslst)
            istimslst = istimslst.reshape(istimsfst.shape)
            iposts = [min(istims[i][-1] + ipost, istims[i][-1] + iisi[i]) for i in np.arange(0, len(self.sweeps))]
            iposts = np.array(iposts)[:, np.newaxis]
            print("tstimsfst: ", tstimsfst, tstimsfst.shape)
            print("tstimslst: ", tstimslst, tstimslst.shape)
            # input('key')
            # ------------
            # holds extracted lag and peak values
        tlags = np.zeros((istimsfst.shape[1]))
        tpeaks = np.zeros((istimsfst.shape[1]))
        ypeaks = np.zeros((istimsfst.shape[1]))
        searchwin = int(0.1 / self.si)  # 0.07
        # ------------
        sweeps = self.sweeps
        ioffset = int(self.vclampsteps[0]["ipost"][-1])
        print(ioffset)
        fh = plt.figure()
        ah = fh.add_subplot(111)
        fh, ah = format_plot(fh, ah, xlab="Time(ms)", ylab="(" + self.channelspecs[0]["units"] + ")", title=self.fid)
        y = self.data[sweeps, iresch, ioffset:].T  # load the full trace
        t = np.arange(0, (y.shape[0] + 2) * self.si, self.si)
        t = t[0:y.shape[0]]
        t = t - t[0]
        t.resize((len(t), 1))
        for i in np.arange(0, len(sweeps)):
            y[:, i] = smooth(y[:, i], windowlen=21, window='hanning')
            m1, c1 = np.linalg.lstsq(np.concatenate((t, np.ones((len(t), 1))), axis=1), y[:, i], rcond=None)[0]
            baseline = (t * m1 + c1)
            y[:, i] = y[:, i] - baseline[:, 0]

        yavg = y.mean(axis=1).T[:, np.newaxis]
        for j in np.arange(0, len(istimsfst[0])):
            ifst = istimsfst[0][j] - ioffset
            ilst = istimslst[0][j] - ioffset
            if (peakdir == '+'):
                ipeaks, _ = find_peaks(yavg[ifst:(ifst + searchwin), 0], prominence=(0.08, None), width=(50, None),
                                       height=(0.2, None))
            if (peakdir == '-'):
                ipeaks, _ = find_peaks(-yavg[ifst:(ifst + searchwin), 0], prominence=(0.08, None), width=(50, None),
                                       height=(0.2, None))
                ah.plot(t[ifst:ilst + ioffset], y[ifst:ilst + ioffset, :], color='gray')
                ah.plot([t[ifst], t[ifst]], [0, 1], 'g')
                ah.plot([t[ilst], t[ilst]], [0, 1], 'r')
            try:
                ipeak = ipeaks[0]
                tpeaks[j] = t[ifst + ipeak, 0]
                tlags[j] = t[ifst + ipeak, 0] - t[ifst, 0]
                ypeaks[j] = yavg[ifst + ipeak, 0]
                ah.plot(tpeaks[j], ypeaks[j], marker='o', markersize=10, color='k')
            except:
                print("!!!!! peaks not found !!!")
                # --------------
        ah.plot(t, yavg)
        ah.set_xlim([t[istimsfst[0][0] - ioffset, 0] - 0.02, t[istimslst[0][-1] - ioffset, 0] + 0.1])
        if (plotdata):
            plt.show()
        return (tlags, ypeaks, fh, ah)

    # ======================================================
    def findpeaks_template_match(self, reschan, trgchan, success_sweeps, figsavepath):
        # template matching algorithm
        # reschan: response channel name
        # trgchan:  stimulus trigger channel name
        # get channel ids for response and trigger channels
        plt.ion()
        fh = plt.figure()
        ah = fh.add_subplot(111)
        ph = []
        # add 4 dummy plots
        # ph = ah.plot(0,0)[0]
        [ph.append(ah.plot(0, 0, color='k')[0]) for i in np.arange(0, 3)]
        fh, ah = plotting_functions.format_plot(fh, ah, xlab="Time (s)", ylab="EPSC (pA)", title="")
        plt.show()
        # --------------------
        if (len(self.sweeps) == 0):
            return ()
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschan, channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchan, channelspec['name'])]
        # -------------
        # get template trace
        # +ve tpre is for time after stimulus onset
        tpreavg = 0.0015
        ipreavg = int(tpreavg / self.si)
        if (self.vclamp < -50):
            tsearch = 0.01
            tpostavg = 0.05
        if (self.vclamp > 10):
            tsearch = 0.02
            tpostavg = 0.2
            yy = self.average_trace(reschan, trgchan, tpreavg, tpostavg, success_sweeps)
            tprestim = tpostavg
            iprestim = int(tprestim / self.si)
            # ------------
            # add null stimulus before the first stimulus
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.sweeps])
        tstims = np.concatenate(((tstims[:, 0] - tprestim)[np.newaxis].reshape((tstims.shape[0], 1)), tstims), axis=1)
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.sweeps])
        istims = np.concatenate(((istims[:, 0] - iprestim)[np.newaxis].reshape((istims.shape[0], 1)), istims), axis=1)
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.sweeps])
        iisi = np.array([int(self.stimprop[sweep]["isi"] / self.si) for sweep in self.sweeps if
                         not np.isnan(self.stimprop[sweep]["isi"])])
        # yy[:,0] = yy[:,0] - yy[0,0]
        # yyn = (yy[:,0] - yy.mean(axis=0))/yy.std(axis=0)
        tt = np.arange(0, (yy.shape[0] + 2) * self.si, self.si)[:, np.newaxis]
        tt = tt[0:yy.shape[0]]  # tt is a column vector
        # np array to hold the individual deconvolved traces for each sweep
        yd = np.zeros((tstims.shape[0], tstims.shape[1], self.samplesize))
        betas = np.zeros(tstims.shape)
        peaks = np.zeros(tstims.shape)
        lags = np.zeros(tstims.shape)
        mins = np.zeros(tstims.shape)
        maxs = np.zeros(tstims.shape)
        mints = np.zeros(tstims.shape)
        maxts = np.zeros(tstims.shape)
        # --------------
        tpoststim = tpostavg + tsearch
        ipoststim = int(tpoststim / self.si)
        sweeps = self.sweeps
        isearch = int(tsearch / self.si)
        for i in np.arange(0, len(sweeps)):
            y = self.data[sweeps[i], iresch, :].T  # load the full trace. y is a column vector
            t = np.arange(0, (y.shape[0] + 2) * self.si, self.si)
            t = t[0:y.shape[0]]
            t.resize((len(t), 1))  # t is a column vector
            # smooth y
            y[:, 0] = smooth(y[:, 0], windowlen=21, window='hanning')
            # detrend y
            m1, c1 = np.linalg.lstsq(np.concatenate((t, np.ones((len(t), 1))), axis=1), y, rcond=None)[0]
            ybase = (t * m1 + c1)
            y = y - ybase
            for j in np.arange(0, len(tstims[i])):
                icorrmax = 0
                if (j == (len(tstims[i]) - 1)):
                    y2 = y[istims[i][j]:(istims[i][j] + (ipoststim + isearch)), :]
                    t2 = t[istims[i][j]:(istims[i][j] + (ipoststim + isearch)), :]
                else:
                    y2 = y[istims[i][j]:istims[i][j + 1] + (isearch), :]
                    t2 = t[istims[i][j]:istims[i][j + 1] + (isearch), :]
                    # -----------
                    # add previous response to template
                    # yyn = yd[i,:,(istims[i][j]+ipre):(istims[i][j]+ipre+len(yy))].sum(axis=0) + yyn
                icorrmax, beta, yyf = template_match(t2, y2, tt, yy)
                betas[i, j] = beta
                imin = np.argmin(yyf * beta)
                imax = np.argmax(yyf * beta)
                if ((self.vclamp < -50) and (beta > 0)):
                    peaks[i, j] = beta * yyf[imin, 0]
                    lags[i, j] = (icorrmax + imin) * self.si
                if ((self.vclamp > 0) and (beta > 0)):
                    peaks[i, j] = beta * yyf[imax, 0]
                    lags[i, j] = (icorrmax + imax) * self.si
                if ((self.vclamp < -50) and (beta < 0)):
                    peaks[i, j] = beta * yyf[imax, 0]
                    lags[i, j] = (icorrmax + imax) * self.si
                if ((self.vclamp > 0) and (beta < 0)):
                    peaks[i, j] = beta * yyf[imin, 0]
                    lags[i, j] = (icorrmax + imin) * self.si
                    # -------------
                    # plt.cla()
                    # plt.clf()
                ah.set_title(self.fid + "_sweep" + str(i) + "_stim" + str(j))
                ph[0].set_xdata(t[istims[i][j]:istims[i][j] + len(yy)])
                ph[0].set_ydata(y[istims[i][j]:istims[i][j] + len(yy)])
                ph[1].set_xdata(t[istims[i][j] + icorrmax:istims[i][j] + icorrmax + len(yy)])
                ph[1].set_ydata(yyf[:, 0] * beta)
                ph[1].set_color('green')
                ph[2].set_xdata(tstims[i][j] + lags[i, j])
                ph[2].set_ydata(peaks[i, j])
                ph[2].set_marker('o')
                ph[2].set_color('k')
                ph[2].set_markersize(10)
                # --------------------------
                for p in ph[0:-1]:
                    xlims = [ah.get_xlim()[0], ah.get_xlim()[1]]
                    ylims = [ah.get_ylim()[0], ah.get_ylim()[1]]
                    if (xlims[0] > np.min(p.get_xdata())) or (xlims[1] < np.max(p.get_xdata())):
                        ah.set_xlim([np.min(p.get_xdata()), np.max(p.get_xdata())])
                    if (ylims[0] > np.min(p.get_ydata())) or (ylims[1] < np.max(p.get_ydata())):
                        ah.set_ylim([np.min(p.get_ydata()), np.max(p.get_ydata())])
                        # --------------
                if (os.path.exists(figsavepath)):
                    figname = self.fid + "_sweep" + str(i) + "_stim" + str(j) + ".png"
                    print("FIG: ", os.path.join(figsavepath, figname))
                    fh.savefig(os.path.join(figsavepath, figname))
                    # --------

        return (lags, betas, peaks)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def template_match_avg(self, reschan, trgchan, peakdir='+'):
        # template matching algorithm
        # fephys: ephys file object
        # tt: t of template
        # yy: y of template
        # tlag: maximum lag to search for correlations, typically 10 ms
        # reschan: response channel name
        # trgchan:  stimulus trigger channel name
        # get channel ids for response and trigger channels
        if (len(self.sweeps) == 0):
            return ()
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschan, channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchan, channelspec['name'])]
        # ---------------
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.sweeps])
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.sweeps])
        tpre = 0.001
        tpost = 0.2
        ipost = int(tpost / self.si)
        tstimsfst = tstims - tpre
        ipre = int(tpre / self.si)
        istimsfst = istims - ipre
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.sweeps])
        iisi = np.array([int(self.stimprop[sweep]["isi"] / self.si) for sweep in self.sweeps if
                         not np.isnan(self.stimprop[sweep]["isi"])])
        # ------------
        if (all(isi) > 0):
            tstimslst = [min(tstims[i][j] + tpost, tstims[i][j] + isi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(tstims[i]))]
        else:
            tstimslst = [max(tstims[i][j] + tpost, tstims[i][j] + isi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(tstims[i]))]
            tstimslst = np.array(tstimslst)
            tstimslst = tstimslst.reshape(tstimsfst.shape)
            tpost = [min(tstims[i][-1] + tpost, tstims[i][-1] + isi[i]) for i in np.arange(0, len(self.sweeps))]
            tpost = np.array(tpost)[:, np.newaxis]
            # -------------
        if (all(isi) > 0):
            istimslst = [min(istims[i][j] + ipost, istims[i][j] + iisi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(istims[i]))]
        else:
            istimslst = [max(istims[i][j] + ipost, istims[i][j] + iisi[i]) for i in np.arange(0, len(self.sweeps)) for j
                         in np.arange(0, len(istims[i]))]
            istimslst = np.array(istimslst)
            istimslst = istimslst.reshape(istimsfst.shape)
            iposts = [min(istims[i][-1] + ipost, istims[i][-1] + iisi[i]) for i in np.arange(0, len(self.sweeps))]
            iposts = np.array(iposts)[:, np.newaxis]
            # =====
        tlags = np.zeros((istimsfst.shape[1]))
        tpeaks = np.zeros((istimsfst.shape[1]))
        ypeaks = np.zeros((istimsfst.shape[1]))
        searchwin = int(0.1 / self.si)  # 0.07
        # ------------
        # holds extracted lag and peak values of the deconvolved trace
        lags = np.zeros(len(self.sweeps))
        tpeaks = np.zeros(len(self.sweeps))
        ypeaks = np.zeros(len(self.sweeps))
        # get template trace
        yy = self.average_trace(reschan, trgchan, tpre)
        yy = yy - yy[0]
        # normalize template so that the correlation is amplitude independent
        yyn = (yy - np.mean(yy)) / np.std(yy)
        # yyn = yyn - yyn[0]
        tt = np.arange(0, (yy.shape[0] + 2) * self.si, self.si)[:, np.newaxis]
        tt = tt[0:yy.shape[0]]
        # reduce yy length
        # yydur = 0.2
        # iyydur = np.where(tt>=yydur)[0][0]
        # tt = tt[0:iyydur,:]
        # yy = yy[0:iyydur]
        # ---------------
        tmaxshift = 0.03
        imaxshift = int(tmaxshift / self.si)
        ioffset = int(self.vclampsteps[0]["ipost"][-1])
        sweeps = self.sweeps
        y = self.data[sweeps, iresch, ioffset:].T  # load the full trace
        t = np.arange(0, (y.shape[0] + 2) * self.si, self.si)
        t = t[0:y.shape[0]]
        t = t - t[0]
        t.resize((len(t), 1))
        for i in np.arange(0, len(sweeps)):
            y[:, i] = smooth(y[:, i], windowlen=21, window='hanning')
            m1, c1 = np.linalg.lstsq(np.concatenate((t, np.ones((len(t), 1))), axis=1), y[:, i], rcond=None)[0]
            baseline = (t * m1 + c1)
            y[:, i] = y[:, i] - baseline[:, 0]
            y = y.mean(axis=1).T
            sy = np.zeros((len(y), 1))  # new y from the shifted and scaled template
            st = np.arange(0, (sy.shape[0]) * self.si, self.si)
            st.resize((len(st), 1))
            dcy = np.zeros(y.shape)  # new y from the shifted and scaled template
            ny = np.zeros(y.shape)  # new y from the shifted and scaled template
            fh = plt.figure()
            ah = fh.add_subplot(111)
            fh, ah = format_plot(fh, ah, xlab="Time(ms)", ylab="(" + self.channelspecs[0]["units"] + ")",
                                 title=self.fid)
        for j in np.arange(0, len(istimsfst[0])):
            print("Sweep: ", i, " Stim: ", j, " istimsfst: ", istimsfst[i][j], " istimslst: ", istimslst[i][j])
            ifst = istimsfst[i][j] - ioffset
            ilst = istimslst[i][j] - ioffset
            corrlen = min((ilst - ifst + 5), yy.shape[0])
            # array to hold the correlations at different time lags
            corrvec = np.zeros(imaxshift)
            # normalize response so that the correlation is amplitude independent
            yn = (y - np.mean(y)) / np.std(y)
            # yn = (y - np.mean(y[(ifst):(ifst+corrlen)]))/np.std(y[(ifst):(ifst+corrlen)])
            yn = yn - yn[ifst]
            print(yn.shape, yyn.shape)
            for k in np.arange(0, imaxshift):
                # perform correlation between the actual trace and the template
                corrvec[k] = np.correlate(yn[(ifst + k):(ifst + corrlen + k)], yyn[0:corrlen, 0])
                # get the index at which the correlation is maximum
            icmax = np.where(corrvec >= np.max(corrvec))[0][0]
            # yysolve = yy
            yysolve = sy[(ifst + corrlen):(ifst + corrlen + yy.shape[0]), 0] + yy[:, 0]
            yysolve.resize((len(yysolve), 1))
            print(y.shape, yy.shape, yysolve.shape, sy.shape, ifst + icmax, ifst + icmax + corrlen, corrlen)
            # # solve normal equation/lstsqr to estimate beta
            beta = np.linalg.lstsq(yysolve, y[ifst + icmax:ifst + icmax + corrlen] - y[ifst], rcond=None)[0]
            # sy[ifst+icmax:ifst+icmax+corrlen,0] = sy[ifst+icmax:ifst+icmax+ corrlen,0] + (yysolve[:,0] * beta) # fit
            sy[ifst + icmax:ifst + icmax + corrlen, 0] = sy[ifst + icmax:ifst + icmax + corrlen, 0] + (
                        yysolve[:, 0] * beta) + y[ifst]  # fit
            dcy[ifst + icmax:ifst + icmax + corrlen] = dcy[ifst + icmax:ifst + icmax + corrlen] + (
                        yy[0:corrlen, 0] * beta)  # deconvolved
            ny[ifst + icmax:ifst + icmax + corrlen] = y[ifst + icmax:ifst + icmax + corrlen]  # part of y that is fit
            # -----------
            if (peakdir == '+'):
                # ipeaks,_ = find_peaks(dcy[ifst:(ifst+searchwin)],prominence=(0.05,None),width=(30,None),height=(0.3,None))
                ipeaks, _ = find_peaks(dcy[ifst + icmax:(ifst + icmax + searchwin)], prominence=(0.08, None),
                                       height=(0.1, None), width=(100, None))
            try:
                ipeak = ipeaks[0]
                tpeaks[j] = st[ifst + ipeak + icmax, 0]
                tlags[j] = st[ifst + ipeak + icmax, 0] - st[ifst, 0]
                ypeaks[j] = dcy[ifst + ipeak + icmax]
                ah.plot(tpeaks[j], ypeaks[j], marker='o', markersize=10, color='k')
            except:
                print("!!!!! peaks not found !!!!!")
                ah.plot([t[ifst], t[ifst]], [0, 1], color='blue', linewidth=2, linestyle='--')
                ah.plot([t[ilst], t[ilst]], [0, 1], color='red', linewidth=2, linestyle='--')
                # ah.plot(tt[0:corrlen]+(icmax*self.si)+t[ifst], (yy[0:corrlen]-yy[0])*beta,'k')
        print('shapes', st.shape, dcy.shape, y.shape)
        ah.plot(t, y, color='grey')
        ah.plot(st, ny, color='black')
        ah.plot(st, sy, color='red')
        ah.plot(st, dcy, color='green')
        ah.set_xlim([t[istimsfst[0][0] - ioffset, 0] - 0.02, t[istimslst[0][-1] - ioffset, 0] + 0.1])
        # ah1.plot(t,s,'red')
        # ah.plot(st,sy,'g')
        return (tlags, ypeaks, fh, ah)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def show(self, ichannels=[], isweeps=[]):
        # display traces
        if (len(isweeps) == 0):
            isweeps = np.arange(0, self.nsweeps)
        if (len(ichannels) == 0):
            ichannels = np.arange(0, self.nchannels)
            # load data if not already loaded
        if (not self.loaddata):
            self.data = self.__loaddata()
        if not (all(isinstance(x, int) for x in ichannels)):
            print("Channel indices must be integers")
            exit()

        t = np.arange(self.tstart, self.tstop, self.si)
        fig = plt.figure(figsize=(10, 8), constrained_layout=False)
        spec = gridspec.GridSpec(nrows=len(ichannels), ncols=1, figure=fig, width_ratios=[1], height_ratios=[4, 1])
        ax = []
        ax.append(fig.add_subplot(spec[0, 0]))
        # ax.append(fig.add_subplot(len(ichannels),1,1))
        pannelcount = 0
        # set figure title
        figtitle = "".join((re.search("(.*/+)(.*)", self.fname)[2], "\n", "#sweeps = ", str(self.nsweeps)))
        ax[0].set_title(figtitle)
        for i in ichannels:
            pannelcount = pannelcount + 1
            if (pannelcount > 1):
                ax.append(fig.add_subplot(spec[pannelcount - 1, 0], sharex=ax[0]))
                # --------------
            channelname = self.channelspecs[i]["name"]
            channelunit = self.channelspecs[i]["units"]
            channelylabel = channelname + ' (' + channelunit + ')'
            ax[pannelcount - 1].set_ylabel(channelylabel)
            for j in isweeps:
                ax[pannelcount - 1].plot(t, self.data[j, i, :])
                # ---------------
        # ------------
        # display average
        ax[0].plot(t, self.data[:, 0, :].mean(axis=0), color='k')
        plt.show()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def __loaddata(self):
        """loads all the sweeps and channels into a numpy ndarray: timeseries * sweeps * channels """
        self.data = np.zeros((self.nsweeps, self.nchannels, self.nsamples))
        # blk = self.reader.read_block(block_index=i,lazy=False,signal_group_mode='split-all',units_group_mode='split-all')
        blk = self.reader.read_block(lazy=False)
        iseg = -1
        for seg in blk.segments:
            iseg = iseg + 1
            isig = -1
            for i, asig in enumerate(seg.analogsignals):
                isig = isig + 1
                asig = asig.magnitude
                self.data[iseg, isig, :] = asig[:, 0]
                # print('Data reading completed')
                # print('Data shape: ',self.data.shape)
        return (self.data)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def info(self):
        channelnames = [channelspec['name'] for channelspec in self.channelspecs]
        channelunits = [channelspec['units'] for channelspec in self.channelspecs]
        print("==================================================================")
        print('fname:', '\t', self.fname)
        print('fid:', '\t', self.fid)
        print('VClamp:', '\t', self.vclamp)
        print('CClamp', '\t', self.cclamp)
        print('nblocks:', '\t', self.nblocks)
        print('nsweeps:', '\t', self.nsweeps)
        print('goodsweeps:\t', self.goodsweeps)
        print('nchannels', '\t', self.nchannels)
        print('channelnames', '\t', channelnames)
        print('channelunits', '\t', channelunits)
        print('sampling interval:', '\t', self.si)
        print('t_start', '\t', self.tstart)
        print('t_stop', '\t', self.tstop)
        print('sample size:', '\t', self.samplesize)
        print('Date:', '\t', self.date)
        print('Time:', '\t', self.time)
        print('Series resistance', '\t', self.sres, ' Mohms')
        print("==================================================================")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def extract_indices_holdingsteps(self, clampch, NstepsPos=1, NstepsNeg=1):
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch, channelspec['name'])]
        t = np.arange(self.tstart, self.tstop, self.si)
        self.iholdsteps = [dict({"neg": np.zeros(NstepsNeg, dtype=np.uint), "pos": np.zeros(NstepsPos, dtype=np.uint)})
                           for sweep in np.arange(0, self.nsweeps)]

        for isweep in np.arange(0, self.nsweeps):
            clamp = self.data[isweep, iclamp, :].T  # [isweep,ichannel,isample]
            dclamp = np.diff(clamp, axis=0)
            dclamp.resize(max(dclamp.shape))
            # find all negative peaks above threshold
            ineg_peaks = find_peaks(-dclamp, height=1, threshold=1)[0][0:NstepsNeg]
            # find all positive peaks above threshold
            ipos_peaks = find_peaks(dclamp, height=1, threshold=1)[0][0:NstepsPos]
            self.iholdsteps[isweep]["neg"] = ineg_peaks
            self.iholdsteps[isweep]["pos"] = ipos_peaks
            # plotting
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t[0:len(dclamp)],dclamp)
            # inegpeaks = self.iholdsteps[isweep]["neg"]
            # ipospeaks = self.iholdsteps[isweep]["pos"]
            # print(inegpeaks,ipospeaks)
            # ah.plot(t[inegpeaks],dclamp[inegpeaks],marker='o',color='r',markersize=10)
            # ah.plot(t[ipospeaks],dclamp[ipospeaks],marker='o',color='g',markersize=10)
            # plt.show()

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def seriesres_voltageclamp(self, resch, clampch):
        # Measure series resistance in a voltage-clamp sweep"
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch, channelspec['name'])]
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch, channelspec['name'])]
        # fh = plt.figure()
        # ah = plt.subplot(111)
        t = np.arange(self.tstart, self.tstop, self.si)
        # istart = np.where(t>=tstart)[0][0]
        # istop = np.where(t>=tstop)[0][0]
        # print(istart,istop)
        for isweep in np.arange(0, self.nsweeps):
            y = self.data[isweep, ires, :].T  # [isweep,ichannel,isample]
            dy = np.diff(y, axis=0)
            dy.resize(max(dy.shape))
            clamp = self.data[isweep, iclamp, :].T  # [isweep,ichannel,isample]
            dclamp = np.diff(clamp, axis=0)
            dclamp.resize(max(dclamp.shape))
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t[:-1],-dclamp)
            # plt.show()
            # find all negative peaks above threshold
            inegpeak_clamp = find_peaks(-dclamp, height=5, threshold=5)[0][0]
            ipospeak_clamp = find_peaks(dclamp, height=5, threshold=5)[0][0]
            deltaclamp = abs(clamp[inegpeak_clamp] - clamp[ipospeak_clamp]) * 1e-3  # in volts
            ipospeak_y = inegpeak_clamp - 20 + np.argmax(y[inegpeak_clamp - 20:ipospeak_clamp + 20])
            inegpeak_y = inegpeak_clamp - 20 + np.argmin(y[inegpeak_clamp - 20:ipospeak_clamp + 20])
            baseliney = np.mean(y[0:inegpeak_y - 20])
            peaky = -(y[inegpeak_y] - baseliney) * 1e-12
            sres = (deltaclamp / peaky) * 1e-6  # in mega ohms
            self.sres[isweep] = sres
            print("deltaclamp: {} deltay: {}, sres: {}".format(deltaclamp, peaky, sres))
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t,clamp)
            # ah.plot(t,y)
            # ah.plot(t[ipospeak_clamp],clamp[ipospeak_clamp],'o','red')
            # ah.plot(t[inegpeak_clamp],clamp[inegpeak_clamp],'o','blue')
            # ah.plot(t[ipospeak_y],y[ipospeak_y],'o','red')
            # ah.plot(t[inegpeak_y],y[inegpeak_y],'o','blue')
            # plt.show()
            # }

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def seriesres_currentclamp(self, resCh, clampCh):
        # Measure series resistance in a current-clamp sweep"
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resCh, channelspec['name'])]
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampCh, channelspec['name'])]
        # fh = plt.figure()
        # ah = plt.subplot(111)
        t = np.arange(self.tstart, self.tstop, self.si)
        # istart = np.where(t>=tstart)[0][0]
        # istop = np.where(t>=tstop)[0][0]
        # print(istart,istop)
        for isweep in np.arange(0, self.nsweeps):
            v = self.data[isweep, ires, :].T  # [isweep,ichannel,isample]
            dv = np.diff(v, axis=0)
            dv.resize(max(dv.shape))
            clamp = self.data[isweep, iclamp, :].T  # [isweep,ichannel,isample]
            dclamp = np.diff(clamp, axis=0)
            dclamp.resize(max(dclamp.shape))
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dclamp, height=10, threshold=10)
            # print("ineg_peaksall:\t",ineg_peaksall)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            ineg_peak = ineg_peaksall[0]
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dclamp, height=10, threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipos_peak = ipos_peaksall[0]
            # print("ipos_peaksall:\t",ipos_peaksall)
            # find the difference between before and after the current jump
            vdel_pos = abs(v[ipos_peak + 8] - v[ipos_peak]) * 1e-3  # in volts
            vdel_neg = abs(v[ineg_peak + 8] - v[ineg_peak]) * 1e-3  # in volts
            idel_pos = abs(clamp[ipos_peak + 1] - clamp[ipos_peak]) * 1e-12  # in amperes
            idel_neg = abs(clamp[ineg_peak + 1] - clamp[ineg_peak]) * 1e-12  # in amperes
            vdel = np.mean([vdel_pos, vdel_neg])
            idel = np.mean([idel_pos, idel_neg])
            sres = (vdel / idel) * 1e-6  # in mega ohms
            self.sres[isweep] = sres
            # print('Series res:\t',self.sres[isweep],' Mega ohms')
            #     ah.plot(t,v)
            #     ah.plot([t[ipos_peak],t[ipos_peak+8]],[v[ipos_peak],v[ipos_peak+8]],color='k')
            #     ah.plot([t[ineg_peak],t[ineg_peak+8]],[v[ineg_peak],v[ineg_peak+8]],color='k')
            # plt.show()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def extract_response(self, reschannel, trgchannel, tpre, tpost, min_isi=0):
        # extract signal from each sweep
        # reschannel: response channel
        # trgchannel: stimulus trigger channel
        # tpre:  time of the start of the response wrt to stimulus time
        # tpost:  time of the stop of the response wrt to stimulus time
        # min_isi: minimum inter-stimulus interval between responses
        # check if isi is <= min_isi
        if ((self.stimprop[0]["nstim"] > 1) and (min_isi > 0)):
            isi = [elm["isi"] for elm in self.stimprop if not np.isnan(elm["isi"])]
            if (len(isi) > 0):
                isi = np.mean(isi)
                if (isi < min_isi):
                    print("isi < min_isi")
                    return ()
                # ---------
        sweeps = self.goodsweeps
        # declare array to hold all the responses
        nsamples = int((tpost - tpre) / self.si)
        res = []
        tt = np.arange(0, (self.tstop - self.tstart), self.si)[np.newaxis].T
        # get channel ids for response and trigger channels
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschannel, channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel, channelspec['name'])]
        lowcut = 0.5
        highcut = 1000
        fs = 1 / self.si
        iholds = self.iholdsteps
        # ---------------
        for isweep in range(0, len(sweeps)):
            tstims = self.stimprop[isweep]["tstims"]
            istims = self.stimprop[isweep]["istims"]
            ipos = iholds[isweep]['pos'][0]
            print(ipos)
            t = tt
            yy = self.data[sweeps[isweep], ires, ipos:].T  # [isweep,ichannel,isample]
            yy = butter_bandpass_filter(yy[:, 0], lowcut, highcut, fs, order=2)[:, np.newaxis]
            m1, c1 = np.linalg.lstsq(np.concatenate((tt, np.ones(tt.shape)), axis=1), yy, rcond=None)[0]
            baseline = (tt * m1 + c1)
            yy = yy - baseline
            print(tt.shape, yy.shape)
            fh = plt.figure()
            ah = fh.add_subplot(111)
            ah.plot(tt[ipos:], yy)
            plt.show()
        #     input()
        #     for i in range(len(istims)):

        #         ifirst = np.where(tt>(tstims[i]+tpre))[0][0]
        #         ilast = np.where(tt>(tstims[i]+tpost))[0][0]

        #         yy = self.data[sweeps[isweep],ires,ifirst:ilast].reshape((ilast-ifirst)) # [isweep,ichannel,isample]
        #         re
        #         input()

        #     # print(tt[ilast:,:].shape,yy.shape)
        #     # yy = smooth(yy,windowlen=21,window='hanning')

        #     print(isweep,tpre,tpost,ipre,ipost,ilast)
        #     yy = yy[(ipre-ilast):(ipost-ilast),0] # [isweep,ichannel,isample]
        #     print(yy.shape)
        #     y[0:len(yy),isweep] = yy - yy[0]
        #     # fh = plt.figure()
        #     # ah = fh.add_subplot(111)
        #     # ah.plot(t,y)
        #     # ah.set_title("sweep: "+str(isweep))
        #     # plt.show()
        #     # plt.close()
        #     # -------------
        # t = tt[ipre:ipre+len(yy)+1]-tstims[0]
        # return(t,y)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def extract_res_props(self, reschannel, trgchannel):
        # Find the peaks and other parameters of the respose (EPSP/EPSC)
        # fh = plt.figure()
        # ah1 = plt.subplot(211)
        # ah2 = plt.subplot(212,sharex=ah1)
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschannel, channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel, channelspec['name'])]
        t = np.arange(self.tstart, self.tstop, self.si)
        # find the trigger pulses
        triggerthres = 500
        tprestim = 0.02
        nprestim = int(tprestim / self.si)
        tpoststim = 0.2
        npoststim = int(tpoststim / self.si)
        for isweep in np.arange(0, self.nsweeps):
            tstims = self.stimprop[isweep]["tstims"]
            istims = self.stimprop[isweep]["istims"]
            ipre = istims[0] - nprestim
            ipost = istims[-1] + npoststim
            yt = self.data[isweep, itrg, ipre:ipost]  # [isweep,ichannel,isample]
            yr = self.data[isweep, ires, ipre:ipost]  # [isweep,ichannel,isample]
            tt = t[ipre:ipost] - t[ipre]
            yr.resize(max(yr.shape))
            yt.resize(max(yt.shape))

    def extract_stim_props(self, trgchannel):
        # get information about stimulation from the trigger channel
        # properties:
        # nstim: number of stimulations
        # istim: index of stimulations
        # tstims: time of each stimulation
        # isi: mean inter stimulus inverval
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel, channelspec['name'])]
        t = np.arange(self.tstart, self.tstop, self.si)
        # find the trigger pulses
        triggerthres = 1000  # 3
        for isweep in np.arange(0, self.nsweeps):
            yt = self.data[isweep, itrg, :]  # [isweep,ichannel,isample]
            yt.resize(self.samplesize)
            istims, _ = find_peaks(yt, height=triggerthres)
            if (len(istims) > 0):
                self.nstim = len(istims)
                self.stimprop[isweep]["nstim"] = self.nstim
                [self.stimprop[isweep]["istims"].append(istim) for istim in istims]
                [self.stimprop[isweep]["tstims"].append(t[istim]) for istim in istims]
                self.stimprop[isweep]["isi"] = 0
                self.isi = 0
                if (len(self.stimprop[isweep]["tstims"]) > 1):
                    self.stimprop[isweep]["isi"] = np.mean(np.diff(self.stimprop[isweep]["tstims"]))
                    # compute average isi
                    self.isi = np.round(np.array([self.stimprop[sweep]["isi"] for sweep in self.goodsweeps]).mean(),
                                        2)  # isi computed from only goodsweeps
                    # ------------------
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def get_clampprop(self, resch, clampch):
        goodsweeps = self.goodsweeps
        iclampch = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch, channelspec['name'])]
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch, channelspec['name'])]
        t = np.arange(self.tstart, self.tstop, self.si)
        yy = self.data[:, iclampch, :].T  # [isweep,ichannel,isample]
        # find the start and end of the current pulse from the clamp currrent
        inegpeaks = np.zeros(self.nsweeps, dtype=np.int)
        ipospeaks = np.zeros(self.nsweeps, dtype=np.int)
        for isweep in np.arange(0, self.nsweeps):
            y = yy[:, 0, isweep]
            dy = np.diff(y, n=1, axis=0)
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dy, height=10, threshold=10)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            inegpeaks[isweep] = ineg_peaksall[np.argmax(dy[ineg_peaksall])] - 1
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dy, height=10, threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipospeaks[isweep] = ipos_peaksall[np.argmax(dy[ipos_peaksall])] - 1
            ipeaksall = np.sort(np.concatenate((ineg_peaksall, ipos_peaksall)))
            [self.clampprop[isweep]["iclamps"].append(iclamp) for iclamp in ipeaksall]
            [self.clampprop[isweep]["tclamps"].append(t[iclamp]) for iclamp in ipeaksall]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def estimate_tau_access_res_from_epsp(self, resch, clampch):
        # estimate the tau and access resistance from charging by fitting a charging equation
        def charging_equation(t, a0, a1, tau):
            return (a0 + a1 * (np.exp(t / tau)))

        # ------------------
        goodsweeps = self.goodsweeps
        iclampch = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch, channelspec['name'])]
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch, channelspec['name'])]
        t = np.arange(self.tstart, self.tstop, self.si)
        yy = self.data[:, iclampch, :].T  # [isweep,ichannel,isample]
        # find the start and end of the current pulse from the clamp currrent
        inegpeaks = np.zeros(self.nsweeps, dtype=np.int)
        ipospeaks = np.zeros(self.nsweeps, dtype=np.int)
        for isweep in np.arange(0, self.nsweeps):
            y = yy[:, 0, isweep]
            dy = np.diff(y, n=1, axis=0)
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dy, height=10, threshold=10)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            inegpeaks[isweep] = ineg_peaksall[np.argmax(dy[ineg_peaksall])] - 1
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dy, height=10, threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipospeaks[isweep] = ipos_peaksall[np.argmax(dy[ipos_peaksall])] - 1
            ipeaksall = np.sort(np.concatenate((ineg_peaksall, ipos_peaksall)))

        # --------------
        avg_step_dur = (t[ipospeaks] - t[inegpeaks]).mean()
        ifitfirst = int(inegpeaks.mean())
        ifitlast = int(ipospeaks.mean())
        istep = yy[:, 0, :].mean(axis=1)[ifitlast] - yy[:, 0, :].mean(axis=1)[ifitfirst]
        iscale = 1e-12
        vscale = 1e-3
        print(istep)
        avg_y = self.data[goodsweeps, iresch, :].T.mean(axis=1)  # avg_y from only goodsweeps
        yc = avg_y[ifitfirst:ifitlast] - avg_y[ifitfirst]
        tc = t[ifitfirst:ifitlast] - t[ifitfirst]
        # perform constrained optimization
        popt, pcov = curve_fit(charging_equation, tc, yc)
        print(popt)
        tau = abs(popt[-1])
        access_res = ((popt[0] + popt[1]) * vscale) / (istep * iscale)
        print("Tau = ", tau)
        print("Access resistance = {} {}".format(access_res / 1e6, "M ohms"))
        # popt, pcov = curve_fit(charging_equation,tc, yc, bounds=([-100,-100,-100], [10, 10, 10]))
        # display the results
        # fh = plt.figure(figsize=(10,8))
        # ah = fh.add_subplot(111)
        # ah.plot(tc,yc)
        # plt.plot(tc, charging_equation(tc, *popt), 'g--')
        # ah.plot(t,yy[:,0,:].mean(axis=-1))
        # ah.plot(t[ifitfirst],yy[:,0,:].mean(axis=1)[ifitfirst],color="red",marker='o',markersize=10)
        # ah.plot(t[ifitlast],yy[:,0,:].mean(axis=1)[ifitlast],color="red",marker='o',markersize=10)
        # plt.show()
        return (tau, access_res)

    def deconvolve_crop_reconvolve(self, resch, clampch, trgch, plot=False):
        # A method to extract individual voltage traces from an EPSP containing multiple short ISIs
        # This method is first described in the paper by Richardson & Silberberg, J Physio., 2008
        def integration_forward_scheme(x0, istimstart, dt, tau, d):
            v = np.zeros((len(d) + 1))
            v[istimstart] = x0
            for i in np.arange(istimstart + 1, len(v)):
                v[i] = v[i - 1] + (dt * (d[i - 1] - v[i - 1]) / tau)

            return (v)

        # ----------------
        def plot_results(t, yy, yreconv, ipeaks, tstims):
            fh = plt.figure(figsize=(8, 5))
            ah = fh.add_subplot(111)
            si = t[1] - t[0]
            prestim = 0.05
            iprestim = int(prestim / si)
            poststim = 0.5
            ipoststim = int(poststim / si)
            for i in np.arange(0, yy.shape[1]):
                itfirst = int(tstims[i, 0] / si) - iprestim
                itlast = int(tstims[i, -1] / si) + ipoststim
                ah.plot(t[itfirst:itlast], yy[itfirst:itlast, i] - baselines[itfirst:itlast, i], color="grey")
                ah.plot(t[itfirst:itlast], yreconv[itfirst:itlast, :, i].sum(axis=1), color="black", linewidth=3)
                for j in range(len(tstims[i])):
                    # ah.plot(t[:-1],ydeconv[:,j,:])
                    ah.plot(t[itfirst:itlast], yreconv[itfirst:itlast, j, i], color="red")
                    # display peaks
                    ah.plot(t[ipeaks[i, j]], yreconv[ipeaks[i, j], j, i], 'o', markersize=5, color="blue")
                    # plot stimulus position
                    ah.plot([tstims[i][j], tstims[i][j]], [0, 1], color="blue")
            return (fh, ah)

        # -------------
        if (len(self.goodsweeps) == 0):  # return if no goodsweeps found
            return ()

        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch, channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgch, channelspec['name'])]
        # ---------------
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.goodsweeps])  # only goodsweeps
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.goodsweeps])  # only goodsweeps
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.goodsweeps])  # only goodsweeps
        iisi = np.array([int(self.stimprop[sweep]["isi"] / self.si) for sweep in self.goodsweeps if
                         not np.isnan(self.stimprop[sweep]["isi"])])  # only goodsweeps
        goodsweeps = self.goodsweeps
        print("goodsweeps: ", goodsweeps)
        # load the full trace
        yy = self.data[goodsweeps, iresch, :].T
        t = np.arange(0, (yy.shape[0] + 2) * self.si, self.si)
        t = t[0:yy.shape[0]]
        t = t - t[0]
        t.resize((len(t), 1))
        si = self.si;
        # tau = self.estimate_tau_access_res_from_epsp(resch,clampch)*2.5
        tau, access_res = self.estimate_tau_access_res_from_epsp(resch, clampch)
        # adjust tau to take into account of tau the synapse
        tau = tau * 1.5
        tclamps = np.array([self.clampprop[goodsweep]["tclamps"] for goodsweep in goodsweeps])
        iclamps = np.array([self.clampprop[goodsweep]["iclamps"] for goodsweep in goodsweeps])
        peaks = np.zeros((len(goodsweeps), len(istims[0])))
        ipeaks = np.zeros((len(goodsweeps), len(istims[0])), dtype=np.int)
        prestimbaselinedur = 0.001
        resdur = 0.017
        ydeconv = np.zeros((len(t) - 1, len(istims[0]), len(goodsweeps)))
        print(ydeconv.shape)
        yreconv = np.zeros((len(t), len(istims[0]), len(goodsweeps)))
        baselines = np.zeros((len(t), len(goodsweeps)))
        sweepcount = 0
        for iy, igoodsweep in zip(np.arange(0, yy.shape[1]), goodsweeps):  # yy has only goodsweeps
            y = smooth(yy[:, iy], windowlen=31, window='hanning')
            # y = yy[:,i]
            ioffset = int(0.1 / self.si)
            t1 = t[iclamps[iy][-1] + ioffset:]
            y1 = y[iclamps[iy][-1] + ioffset:]
            # print(ioffset)
            # input()
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t1,y1)
            # plt.show()
            # m1,c1 = np.linalg.lstsq(np.concatenate((t1,np.ones((len(t1),1))),axis=1),y1,rcond=None)[0]
            # baselines[:,i] = (t*m1 + c1)[:,0]
            m2, m1, c1 = \
            np.linalg.lstsq(np.concatenate((pow(t1, 1.5), t1, np.ones((len(t1), 1))), axis=1), y1, rcond=None)[0]
            baselines[:, iy] = (pow(t, 1.5) * m2 + t * m1 + c1)[:, 0]
            y = y - baselines[:, iy]
            dy = np.diff(y, n=1)
            d = ((dy / si) * tau) + y[0:-1]
            d = smooth(d, windowlen=7, window='hanning')
            # crop the individual responses
            for j in range(0, len(istims[iy])):
                print("iy: {},igoodsweep: {},j: {}".format(iy, igoodsweep, j))
                isstart = istims[iy][j] + int(prestimbaselinedur / si)
                isstop = istims[iy][j] + int(resdur / si)
                ydeconv[isstart:isstop, j, iy] = d[isstart:isstop]
                yreconv[:, j, iy] = integration_forward_scheme(0, isstart, si, tau, ydeconv[:, j, iy])
                ymax = max(np.abs(yreconv[:, j, iy]))
                ipeaksall, _ = find_peaks(np.abs(yreconv[:, j, iy]), prominence=ymax / 2)
                ipeaks[iy, j] = ipeaksall[0]
                peaks[iy, j] = yreconv[ipeaks[iy, j], j, iy]
                # ----------------------
        fh, ah = plot_results(t, yy, yreconv, ipeaks, tstims)
        return (peaks, fh, ah)

    def __del__(self):
        print("Object of EphysClass has been deleted")
        pass


