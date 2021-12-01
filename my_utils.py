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
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import multiprocessing
import numpy as np
import copy
from scipy.fftpack import fft, ifft
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from Artifact_Detection import *
from model.model import *
from model.LSTM_FCN import *


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


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def frag_check_multi_show(signal, start_point=0, win_count=0):  # show_data既可以加[]也可以不加
    if win_count > 99:  # 限制窗口数量，防止太长的无效或大体动片段，导致卡死
        win_count = 99
    line = int(win_count ** 0.5) if (int(win_count ** 0.5)) ** 2 == win_count else int(win_count ** 0.5) + 1
    list = int(win_count ** 0.5) if (int(win_count ** 0.5)) * line >= win_count else int(win_count ** 0.5) + 1

    plt.figure(figsize=(16, 10))
    for i in range(win_count):
        plt.subplot(line, list, i + 1)  # 整体加多一行，list+i是因为第一整行占了全部列，所以下一个子图在此基础上+1开始算
        # plt.ylim(0, 0.02)
        if signal[start_point + i][0] == 0:
            plt.plot(signal[start_point + i, 1:], color='green', label="正常数据")
        if signal[start_point + i][0] == 1:
            plt.plot(signal[start_point + i, 1:], color='red', label="大体动")
        if signal[start_point + i][0] == 2:
            plt.plot(signal[start_point + i, 1:], color='blue', label="小体动")
        if signal[start_point + i][0] == 3:
            plt.plot(signal[start_point + i, 1:], color='Yellow', label="深呼吸")
        if signal[start_point + i][0] == 4:
            plt.plot(signal[start_point + i, 1:], color='purple', label="脉冲体动")
        if signal[start_point + i][0] == 5:
            plt.plot(signal[start_point + i, 1:], color='orange', label="无效片段")
    plt.show()


def signal_split_meth1(data_input, split_step=1, split_len=5, sample_rate=100):
    begin_point = 0
    frag_len = split_len * sample_rate

    frag_data_ch0 = data_input[begin_point:begin_point + frag_len]  #

    begin_point += split_step * sample_rate

    while begin_point + frag_len < len(data_input):
        frag_data_ch0 = np.vstack((frag_data_ch0, data_input[begin_point: begin_point + frag_len]))
        begin_point += split_step * sample_rate
        print('begin_point is : %d / %d' % (begin_point, len(data_input)))

    # frag_data_ch0 = np.vstack((frag_data_ch0, data_input[-frag_len:])) #把末尾一段单独分割，但标签输出和显示比较麻烦，暂时注释
    tem_data = copy.deepcopy(frag_data_ch0)
    frag_data_ch1 = np.zeros_like(frag_data_ch0)
    # for i in tqdm(range(len(frag_data_ch1))):
    #     frag_data_ch1[i] = Butterworth(frag_data_ch0[i], type='lowpass', lowcut=1, order=2, Sample_org=100)
    # for i in tqdm(range(len(frag_data_ch0))):
    #     frag_data_ch0[i] = Butterworth(frag_data_ch0[i], type='bandpass', lowcut=2, highcut=15, order=2, Sample_org=100)

    frag_data_ch0 = frag_data_ch0.reshape(frag_data_ch0.shape[0], 1, frag_data_ch0.shape[1])
    frag_data_ch1 = frag_data_ch1.reshape(frag_data_ch1.shape[0], 1, frag_data_ch1.shape[1])

    # for i in tqdm(range(len(frag_data_ch0)), desc="prep_data_ch0 min_max normalize : "):
    #     frag_data_ch0[i] = preprocessing.MinMaxScaler().fit_transform(frag_data_ch0[i].reshape(-1, 1)).reshape(1, -1)
    # for i in tqdm(range(len(frag_data_ch1)), desc="prep_data_ch1 min_max normalize : "):
    #     frag_data_ch1[i] = preprocessing.MinMaxScaler().fit_transform(frag_data_ch1[i].reshape(-1, 1)).reshape(1, -1)

    # for i in range(len(frag_data_ch0)):
    #     plt.plot(frag_data_ch0[i, 0, :], color='blue', label="Ch0")
    #     plt.plot(frag_data_ch1[i, 0, :] - 1750, color='red', label="Ch1")
    #     plt.plot(tem_data[i, :] - 1950, color='green', label="org")
    #     plt.show()

    frag_data = np.concatenate((frag_data_ch0, frag_data_ch1), axis=1)

    print(frag_data.shape)
    return frag_data_ch0


def signal_split_meth2(data_input, split_len=10, sample_rate=100):
    begin_point = 0
    frag_len = split_len * sample_rate

    frag_data_ch0 = np.full([len(data_input)//frag_len,frag_len],np.nan)

    for i in trange(len(data_input)//frag_len):
        frag_data_ch0[i] = data_input[begin_point: begin_point + frag_len]
        begin_point += frag_len

    frag_data_ch1 = np.zeros_like(frag_data_ch0)
    # for i in tqdm(range(len(frag_data_ch1))):
    #     frag_data_ch1[i] = Butterworth(frag_data_ch0[i], type='lowpass', lowcut=1, order=2, Sample_org=100)
    # for i in tqdm(range(len(frag_data_ch0))):
    #     frag_data_ch0[i] = Butterworth(frag_data_ch0[i], type='bandpass', lowcut=2, highcut=15, order=2, Sample_org=100)

    frag_data_ch0 = frag_data_ch0.reshape(frag_data_ch0.shape[0], 1, frag_data_ch0.shape[1])
    frag_data_ch1 = frag_data_ch1.reshape(frag_data_ch1.shape[0], 1, frag_data_ch1.shape[1])

    frag_data = np.concatenate((frag_data_ch0, frag_data_ch1), axis=1)

    print('The shape of frag_data after split is : ',frag_data.shape)
    return frag_data_ch0


def org_artifact_show(file_Path, down_sample_rate=10, start_point=0, show_len=0, Y_shift=0):
    if os.path.exists(os.path.join(file_Path, "raw_org.txt")) and os.path.exists(
            os.path.join(file_Path, "Artifact_a.txt")):
        data_path = os.path.join(file_Path, "raw_org.txt")  # 原始数据路径,"\\raw_org.txt"也可以
        label_path = os.path.join(file_Path, "Artifact_a.txt")  # 体动数据路径
    elif os.path.exists(os.path.join(file_Path, "new_org_1000hz.txt")) and os.path.exists(
            os.path.join(file_Path, "Artifact_a.txt")):
        data_path = os.path.join(file_Path, "new_org_1000hz.txt")  # 原始数据路径,"\\raw_org.txt"也可以
        label_path = os.path.join(file_Path, "Artifact_a.txt")  # 体动数据路径
    else:
        print('数据或标签打开错误，不存在！')

    orgBCG = pd.read_csv(data_path, header=None).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
    orgLabel = pd.read_csv(label_path, header=None).to_numpy().reshape(-1, 4)  # 标签数据读取为numpy形式，并reshape为n行4列的数组

    orgBCG, orgLabel = data_Subsampled(orgBCG=orgBCG, orgLabel=orgLabel, sampleNum=down_sample_rate)
    newBCG = Butterworth(orgBCG, type='lowpass', lowcut=15, order=2, Sample_org=1000 / down_sample_rate)
    temBCG = copy.deepcopy(newBCG)

    ArtifactData_tpye0 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye1 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye2 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye3 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye4 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye5 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组

    for i in range(orgLabel.shape[0]):
        if orgLabel[i][1] == 1:  # 根据体动类型分别判断和赋值
            ArtifactData_tpye1[orgLabel[i][2]:orgLabel[i][3]] = newBCG[orgLabel[i][2]:orgLabel[i][3]]
        elif orgLabel[i][1] == 2:
            ArtifactData_tpye2[orgLabel[i][2]:orgLabel[i][3]] = newBCG[orgLabel[i][2]:orgLabel[i][3]]
        elif orgLabel[i][1] == 3:
            ArtifactData_tpye3[orgLabel[i][2]:orgLabel[i][3]] = newBCG[orgLabel[i][2]:orgLabel[i][3]]
        elif orgLabel[i][1] == 4:
            ArtifactData_tpye4[orgLabel[i][2]:orgLabel[i][3]] = newBCG[orgLabel[i][2]:orgLabel[i][3]]
        elif orgLabel[i][1] == 5:
            ArtifactData_tpye5[orgLabel[i][2]:orgLabel[i][3]] = newBCG[orgLabel[i][2]:orgLabel[i][3]]
        else:
            print('标签类型错误')
        temBCG[orgLabel[i][2]:orgLabel[i][3]] = ArtifactData_tpye0[orgLabel[i][2]:orgLabel[i][3]]
    ArtifactData_tpye0 = copy.deepcopy(temBCG)

    plt.plot(ArtifactData_tpye0[start_point:start_point + show_len] + Y_shift, color='green', )
    plt.plot(ArtifactData_tpye1[start_point:start_point + show_len] + Y_shift, color='red', )
    plt.plot(ArtifactData_tpye2[start_point:start_point + show_len] + Y_shift, color='blue', )
    plt.plot(ArtifactData_tpye3[start_point:start_point + show_len] + Y_shift, color='Gold', )
    plt.plot(ArtifactData_tpye4[start_point:start_point + show_len] + Y_shift, color='purple', )
    plt.plot(ArtifactData_tpye5[start_point:start_point + show_len] + Y_shift, color='orange', )


def statistics_show(label_true, label_pred, label_prob):
    plt.subplot(1, 2, 1)
    confusion = confusion_matrix(label_true, label_pred)
    print(confusion)
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[second_index][first_index])

    plt.subplot(1, 2, 2)
    # 二分类　ＲＯＣ曲线
    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）
    fpr, tpr, thresholds = roc_curve(label_true, label_prob)
    # auc = auc(fpr, tpr)   #auc
    auc_value = auc(fpr, tpr)
    print("AUC : ", auc_value)
    # plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='VGG16 (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.savefig("../images/ROC/ROC_2分类.png")
    plt.show()
