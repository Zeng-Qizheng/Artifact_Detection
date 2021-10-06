# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import sys

from scipy import signal
from glob import glob

from SQassess.SQIassessment import *
from Preprocessing import BCG_Operation

def Butterworth(x,type,lowcut = 0,highcut = 0,order = 10):
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
    if type == "lowpass" :       #低通滤波处理
        b, a = signal.butter(order, lowcut/(Sample_org*0.5), btype='lowpass')
        return signal.filtfilt(b, a, np.array(x))
    elif type == "bandpass":     #带通滤波处理
        low = lowcut/(Sample_org*0.5)
        high = highcut/(Sample_org*0.5)
        b, a = signal.butter(order, [low,high], btype='bandpass')
        return signal.filtfilt(b, a, np.array(x))
    elif type == "highpass" :    #高通滤波处理
        b, a = signal.butter(order, highcut/(Sample_org*0.5), btype='highpass')
        return signal.filtfilt(b, a, np.array(x))
    else :                       #警告,滤波器类型必须有
        print("Please choose a type of fliter")

def distEuclidean(veca,vecb):
    """
    计算欧几里得距离
    """
    return np.sqrt( np.sum( np.square(veca-vecb) ) )

def InitBeatDetect(data):
    """
    函数说明：
    初始查找合理心跳点
    :param data:                  输入数据信号
    :param maxi_index:            输入数据峰值坐标
    :return:                      处理后的峰值坐标
    """
    length = len(data)
    InitBeat = np.array([])
    win_min = 0
    win_max = 1000
    while (True):
        beat = int(np.argmax( data[win_min:win_max] ) + win_min)
        InitBeat = np.append(InitBeat, beat)
        win_min = max(0, beat + 500)
        win_max = min(length, beat + 1500)
        if (win_min >= length):
            break
    InitBeat = InitBeat.astype(int)
    print(InitBeat[-1])
    if data[InitBeat[-1]] < 0.8*data[InitBeat[-2]] :
        InitBeat = np.delete(InitBeat, -1)
    return InitBeat

def Modeldetect(data,ModelLength,Jpeak,ECG=[]):
    """
    函数说明：对信号data进行模板检测。检通过选取每段之间的最大值为疑似J峰，然后相加平均形成模板
    :param data:                     输入待检测模板信号
    :param ModelLength:              输入模板长度
    :param Jpeal:                    输入预设J峰值
    :return:                         返还模板信号
    """
    test = []
    for peak in Jpeak:
        if peak < ModelLength / 2 or (peak + ModelLength) > len(data):
            continue
        else:
            test.append(data[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))])
    meanBCG = np.zeros(ModelLength)        # ----------------------对初始预判J峰的信号段相加平均
    for num in range(len(test)):
        meanBCG += test[num]
    meanBCG = meanBCG / len(test)
    dit = np.array([])                     # ----------------------计算初始预判信号与平均信号的相似性
    for num in range(len(test)):
        dit = np.append(dit, distEuclidean(test[num], meanBCG) * 1)

    indexmin = np.array([])                # -----------------------选择与平均信号最相似的2个原始信号
    for num in range(7):
        if len(dit)>1 :
            indexmin = np.append( indexmin, np.argmin(dit) )
            dit[np.argmin(dit)] = float("inf")
        else:
            pass
    indexmin = indexmin.astype(int)
    Model = np.zeros(ModelLength)

    for num in indexmin:
        Model += test[num]
    Model = Model/7
    return Model

def CorBeatDetection(data, model):
    """
    函数说明：计算模板和BCG信号的相关函数，通过相关函数来定位出最后的心跳位置
    :param data:               输入BCG信号
    :param model:              输入模板信号
    :return:                   返回心跳位置
    """
    BCGcor = np.correlate(data, model, "same")
    return InitBeatDetect(BCGcor)



Sample_org = 1000
ecg_dir = r"D:\Studysoftware\SQI\new_data\ecg*.txt"
bcg_dir = r"D:\Studysoftware\SQI\new_data\bcg*.txt"

bcg_list = glob(bcg_dir)

# 建立各种信息数组
ID = []
b_SQI = []
t_SQI = []
i_SQI = []
a_SQI = []
s_SQI = []
k_SQI = []

for file_dir in bcg_list :
    bcg = pd.read_csv(file_dir, delimiter='\t', header=None).to_numpy().reshape(-1)
    # 带通滤波(2~20Hz)
    bcg = BCG_Operation(sample_rate=1000).Butterworth(bcg, "bandpass", lowcut=2, highcut=20, order=2)
    # Method1
    beat1 = InitBeatDetect(bcg)
    # Method2
    model = Modeldetect(bcg, 700, beat1)
    beat2 = CorBeatDetection(bcg, model)
    # SQI
    ID.append(file_dir[30: -4])
    b_SQI.append( bSQI(beat1, beat2) )
    t_SQI.append( tSQI(bcg, beat2) )
    i_SQI.append( iSQI(beat2) )
    a_SQI.append( aSQI(bcg, 100) )
    s_SQI.append( sSQI(bcg)[0] )
    k_SQI.append( sSQI(bcg)[1] )

pd.DataFrame(np.array([ID,b_SQI,t_SQI,i_SQI,a_SQI,s_SQI,k_SQI]).T,
             columns=["ID","bSQI","tSQI","iSQI","aSQI","sSQI","kSQI"]).to_csv("./comb.csv", index=False)

sys.exit()



ecg = pd.read_csv(ecg_dir, delimiter='\t', header=None).to_numpy().reshape(-1)
bcg = pd.read_csv(bcg_dir, delimiter='\t', header=None).to_numpy().reshape(-1)
print(ecg.shape, bcg.shape)

bcg = Butterworth(bcg, "bandpass", lowcut=2, highcut=20, order=2)
beat1 = InitBeatDetect(bcg)
model = Modeldetect(bcg, 700, beat1)
beat2 = CorBeatDetection(bcg, model)

print(beat2.shape)
print(tSQI(bcg, beat2))
print(iSQI(beat2))
print(aSQI(bcg, 100))
print(sSQI(bcg))
plt.figure()
# plt.plot(ecg)
plt.plot(bcg)
plt.plot(beat1, bcg[beat1], 'r.')
plt.plot(beat2, bcg[beat2], 'b*')
plt.show()