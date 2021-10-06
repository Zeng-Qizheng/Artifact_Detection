# encoding:utf-8

"""
@ date:             2020-09-16
@ author:           jingxian
@ illustration:     Pre-processing
"""


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
from scipy import signal
from scipy import fftpack


def Dilate(x, N, g, M):
    returndata = np.array([])
    for num in range(N - M + 1):
        returndata = np.append(returndata, np.min(np.array(x[num:num + M]) - np.array(g)))
    return returndata


def Eorde(x, N, g, M):
    returndata = np.array([])
    for num in range(N - M + 1):
        returndata = np.append(returndata, np.max(np.array(x[num:num + M]) - np.array(g)))
    return returndata


def fin_turn(data, peak):
    if len(data) == 0 or len(peak) == 0: return peak
    return_peak = []
    for p in peak:
        minx, maxx = max(0, p - 100), min(len(data), p + 100)
        return_peak.append(minx + np.argmax(data[minx: maxx]))
    return return_peak


class BCG_Operation():
    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate

    def down_sample(self, data=None, down_radio=10):
        if data is None:
            raise ValueError("data is None, please given an real value!")
        data = data[:len(data) // down_radio * down_radio].reshape(-1, down_radio)[:, 0]
        self.sample_rate = self.sample_rate / down_radio
        return data

    def Splitwin(self, data=None, len_win=None, coverage=1.0, calculate_to_end=False):
        """
        分窗
        :param len_win:  length of window
        :return:         signal windows
        """
        if (len_win is None) or (data is None):
            raise ValueError("length of window or data is None, please given an real value!")
        else:
            length = len_win * self.sample_rate  # number point of a window
        # step of split windows
        step = length * coverage
        start = 0
        Splitdata = []
        while (len(data) - start >= length):
            Splitdata.append(data[int(start):int(start + length)])
            start += step
        if calculate_to_end and (len(data) - start > 2000):
            remain = len(data) - start
            start = start - step
            step = int(remain / 2000)
            start = start + step * 2000
            Splitdata.append(data[int(start):int(start + length)])
            return np.array(Splitdata), step
        elif calculate_to_end:
            return np.array(Splitdata), 0
        else:
            return np.array(Splitdata)

    def Butterworth(self, data, type, low_cut=0.0, high_cut=0.0, order=10):
        """
        :param type:      Type of Butter. filter, lowpass, bandpass, ...
        :param lowcut:    Low cutoff frequency
        :param highcut:   High cutoff frequency
        :param order:     Order of filter
        :return:          Signal after filtering
        """
        if type == "lowpass":  # 低通滤波处理
            b, a = signal.butter(order, low_cut / (self.sample_rate * 0.5), btype='lowpass')
            return signal.filtfilt(b, a, np.array(data))
        elif type == "bandpass":  # 带通滤波处理
            low = low_cut / (self.sample_rate * 0.5)
            high = high_cut / (self.sample_rate * 0.5)
            b, a = signal.butter(order, [low, high], btype='bandpass')
            return signal.filtfilt(b, a, np.array(data))
        elif type == "highpass":  # 高通滤波处理
            b, a = signal.butter(order, high_cut / (self.sample_rate * 0.5), btype='highpass')
            return signal.filtfilt(b, a, np.array(data))
        else:  # 警告,滤波器类型必须有
            raise ValueError("Please choose a type of fliter")

    def MorphologicalFilter(self, data=None, M=200, get_bre=False):
        """
        :param data:         Input signal
        :param M:            Length of structural element
        :return:             Signal after filter
        """
        if not data.any():
            raise ValueError("The input data is None, please given real value data")
        g = np.ones(M)
        Data_pre = np.insert(data, 0, np.zeros(M))
        Data_pre = np.insert(Data_pre, -1, np.zeros(M))
        # Opening: 腐蚀 + 膨胀
        out1 = Eorde(Data_pre, len(Data_pre), g, M)
        out2 = Dilate(out1, len(out1), g, M)
        out2 = np.insert(out2, 0, np.zeros(M - 2))
        # Closing: 膨胀 + 腐蚀
        out5 = Dilate(Data_pre, len(Data_pre), g, M)
        out6 = Eorde(out5, len(out5), g, M)
        out6 = np.insert(out6, 0, np.zeros(M - 2))

        baseline = (out2 + out6) / 2
        # -------------------------保留剩余价值------------------------
        data_filtered = Data_pre[:len(baseline)] - baseline
        data_filtered = data_filtered[M: M + len(data)]
        baseline = baseline[M:]
        data_filtered[-1] = data_filtered[-2] = data_filtered[-3]
        baseline[-1] = baseline[-2] = baseline[-3]
        if get_bre:
            return data_filtered, baseline
        else:
            return data_filtered

    def Iirnotch(self, data=None, cut_fre=50, quality=3):
        """陷波器"""
        b, a = signal.iirnotch(cut_fre / (self.sample_rate * 0.5), quality)
        return signal.filtfilt(b, a, np.array(data))

    def ChebyFilter(self, data, rp=1, type=None, low_cut=0, high_cut=0, order=10):
        """
        切比雪夫滤波器
        :param data:              Input signal
        :param rp:                The maximum ripple allowed
        :param type:              'lowpass', 'bandpass, 'highpass'
        :param low_cut:           Low cut-off fre
        :param high_cut:          High cut-off fre
        :param order:             The order of filter
        :return:                  Signal after filter
        """
        if type == 'lowpass':
            b, a = signal.cheby1(order, rp, low_cut, btype='lowpass', fs=self.sample_rate)
            return signal.filtfilt(b, a, np.array(data))
        elif type == 'bandpass':
            b, a = signal.cheby1(order, rp, [low_cut, high_cut], btype='bandpass', fs=self.sample_rate)
            return signal.filtfilt(b, a, np.array(data))
        elif type == 'highpass':
            b, a = signal.cheby1(order, rp, high_cut, btype='highpass', fs=self.sample_rate)
            return signal.filtfilt(b, a, np.array(data))
        else:
            raise ValueError("The type of filter is None, please given the real value!")

    def Envelope(self, data):
        """取信号包络"""
        if len(data) <= 1: raise ValueError("Wrong input data")
        hx = fftpack.hilbert(data)
        return np.sqrt(hx ** 2, data ** 2)

    def wavelet_trans(self, data,c_level=['aaa','aad'], wavelet='db4', mode='symmetric',maxlevel=10):
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode=mode)
        for c in c_level :
            new_wp[c] = wp[c]
        return new_wp.reconstruct()

    def em_decomposition(self, data):
        from pyhht.emd import EMD
        return EMD(data).decompose()

