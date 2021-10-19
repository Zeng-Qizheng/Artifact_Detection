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
from matplotlib.pylab import mpl
import multiprocessing
from model import vgg
import numpy as np
from scipy.fftpack import fft, ifft

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def my_fft1(signal, sampling_point=250):
    """
    Author:Qz
    函数说明:对输入信号进行FFT，并画出频谱图
    :param signal:                输入原始信号
    :param sampling_point:        每秒的采样点，大于信号最高频率的两倍，奈奎斯特采样定理
    :return:                      无
    """
    # 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，
    # 所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    # x = np.linspace(0, 1, sampling_point)

    # 设置需要采样的信号，频率分量有200，400和600
    # y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)
    fft_y = fft(signal)  # 快速傅里叶变换

    # N = 1400
    x = np.arange(len(fft_y))  # 频率个数
    half_x = x[range(int(len(fft_y) / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / len(fft_y)  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(len(fft_y) / 2))]  # 由于对称性，只取一半区间（单边频谱）

    # plt.subplot(231)
    # plt.plot(x, y)
    # plt.title('原始波形')

    plt.subplot(232)
    plt.plot(x, fft_y, 'black')
    plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

    plt.subplot(233)
    plt.plot(x, abs_y, 'r')
    plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

    plt.subplot(234)
    plt.plot(x, angle_y, 'violet')
    plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

    plt.subplot(235)
    plt.plot(x, normalization_y, 'g')
    plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

    plt.subplot(236)
    plt.plot(half_x, normalization_half_y, 'blue')
    plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')

    plt.show()


def my_fft2(signal, fs=100):
    """
    Author:Qz
    函数说明:对输入信号进行FFT，并画出频谱图（目前这个版本效果最好，最大问题在于FFT频谱分析原理不太懂，导致移植困难，频率分辨率不是很懂）
    :param signal:                输入原始信号
    :param fs:                    信号频率？
    :return:                      无
    """
    x = range(len(signal))  # X轴刻度
    delta_f = 1 * fs / len(signal)  # 频率分辨率，为1*fs/N

    fft_aC1 = fft(signal)  # 快速傅里叶变换

    aC1xf = np.arange(len(signal))  # 频率
    aC1xf2 = aC1xf[range(int(len(x) / 2))]  # 取一半区间

    aC1yf = abs(fft(signal))  # 取绝对值
    aC1yf1 = abs(fft(signal)) / len(x)  # 归一化处理
    aC1yf2 = aC1yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Original Wave')

    plt.subplot(2, 1, 2)
    plt.ylim(0, 10)
    plt.plot(aC1xf2 * delta_f, aC1yf2, 'black')
    plt.title('FFT', fontsize=10, color='black')
    plt.show()


def my_fft3(signal):
    """
    Author:Qz
    函数说明:对输入信号进行FFT，并画出频谱图
    :param signal:                输入原始信号
    :param sampling_point:        每秒的采样点，大于信号最高频率的两倍，奈奎斯特采样定理
    :return:                      无
    """
    # 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）

    # 设置需要采样的信号，频率分量有180，390和600

    yy = fft(signal)  # 快速傅里叶变换

    yreal = yy.real  # 获取实数部分
    yimag = yy.imag  # 获取虚数部分

    yf = abs(fft(signal))  # 取绝对值
    yf1 = abs(fft(signal)) / len(signal)  # 归一化处理
    yf2 = yf1[range(int(len(signal) / 2))]  # 由于对称性，只取一半区间

    xf = np.arange(len(signal))  # 频率
    xf1 = xf
    xf2 = xf[range(int(len(signal) / 2))]  # 取一半区间

    plt.subplot(221)
    plt.plot(signal)
    plt.title('Original wave')

    plt.subplot(222)
    plt.plot(xf, yf, 'r')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表

    plt.subplot(223)
    plt.plot(xf1, yf1, 'g')
    plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

    plt.subplot(224)
    plt.plot(xf2, yf2, 'b')
    plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

    plt.show()


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
