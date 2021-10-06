import os
import json
import random
from scipy import signal
from scipy import fftpack
import numba as nb
import numpy as np
import torch
import datetime
from os import times
import time
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from model import vgg


def sample_rate_change(bcg_data, change_rate):
    """
    Author:Qz
    函数说明:对原时间序列进行降采样,不用降采样是最快的，3s多一点跑完程序，十倍降采样变成3.9s，开启且只有1倍降采样要12s多
    :param sampleNum:             输入降采样倍数
    :return:                      无
    """
    if change_rate < 0:
        temBCG = np.full(len(bcg_data) // (-change_rate), np.nan)  # 创建与orgBCG降采样后一样长度的空数组
        for i in range(len(bcg_data) // (-change_rate)):
            temBCG[i] = bcg_data[i * (-change_rate)]
    if change_rate > 0:
        pass  # load...

    return temBCG


def Butterworth(x, type, lowcut=0, highcut=0, order=10, Sample_org=1000):
    """
    函数说明：
    将输入信号x，经过一Butterworth滤波器后，输出信号
    :param x:                        输入处理信号
    :param type:                     滤波类型(lowpass,highpass,bandpass)
    :param lowcut:                   低频带截止频率
    :param highcut:                  高频带截止频率
    :param order:                    滤波器阶数
    :return:                         返还处理后的信号
    """
    if type == "lowpass":  # 低通滤波处理
        b, a = signal.butter(order, lowcut / (Sample_org * 0.5), btype='lowpass')
        return signal.filtfilt(b, a, np.array(x))
    elif type == "bandpass":  # 带通滤波处理
        low = lowcut / (Sample_org * 0.5)
        high = highcut / (Sample_org * 0.5)
        b, a = signal.butter(order, [low, high], btype='bandpass')
        return signal.filtfilt(b, a, np.array(x))
    elif type == "highpass":  # 高通滤波处理
        b, a = signal.butter(order, highcut / (Sample_org * 0.5), btype='highpass')
        return signal.filtfilt(b, a, np.array(x))
    else:  # 警告,滤波器类型必须有
        print("Please choose a type of fliter")


def signal_split(data_input, split_step=1, split_len=5, sample_rate=100):
    begin_point = 0
    frag_len = split_len * sample_rate

    frag_data = data_input[begin_point:begin_point + frag_len]  #

    begin_point += split_step * sample_rate

    while begin_point + frag_len < len(data_input):
        frag_data = np.vstack((frag_data, data_input[begin_point: begin_point + frag_len]))
        begin_point += split_step * sample_rate
        print('begin_point is : %d / %d' % (begin_point, len(data_input)))

    frag_data = np.vstack((frag_data, data_input[-frag_len:]))  #

    print(frag_data.shape)
    return frag_data
