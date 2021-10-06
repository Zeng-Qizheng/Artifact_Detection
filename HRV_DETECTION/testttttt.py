# _*_ coding: utf-8 _*_
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import datetime
import Investiage

from scipy import signal
from scipy import fftpack
from Preprocessing import BCG_Operation
from TemplateMatch import TemplateMarching,Cor,Dist
from scipy import stats
import pyCompare
# import pingouin as pg

def central_interval(data):
    """
    输入信号---->带通滤波---->模板匹配----->希尔伯特变换取包络----->获取包络最大能量对应频率
    :param data:                    输入信号
    :return:                        检测心跳频率对应间期
    """
    # 带通滤波
    operation = BCG_Operation(sample_rate=1000)
    BCG = operation.Butterworth(data, type="bandpass", low_cut=1, high_cut=10, order=2)
    # 模板匹配
    tempplate = TemplateMarching(BCG, samplate_rate=1000)
    initPeak = tempplate.getInitPeak().astype(int)
    model = tempplate.get_template(initPeak, num=3)
    BCG_cor = Cor(BCG, model)
    # 希尔伯特变换取包络
    hx = fftpack.hilbert(BCG_cor)
    hy = np.sqrt(BCG_cor ** 2 + hx ** 2)
    hy = operation.Butterworth(hy, type="bandpass", low_cut=0.1, high_cut=3, order=2)
    # 获取包络最大能量对应频率
    hy_fft = np.fft.fft(hy, len(hy))
    hy_fft = hy_fft * np.conj(hy_fft) /1000
    index = 5 + np.argmax(hy_fft[5:len(hy_fft)//2])
    return 1000/(0.1*index)

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd  = np.std(diff, axis=0)            # Standard deviation of the difference
    limitOfAgreementRange = (md + (1.96 * sd)) - (md - 1.96 * sd)
    offset = (limitOfAgreementRange / 100.0) * 1.5
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='red', linestyle='-')
    plt.axhline(0, color='gray', linestyle='-')
    plt.text(np.max(mean), md + offset, 'Mean', ha="right", va="bottom")
    plt.text(np.max(mean), md - offset,  f'{md:.2f}', ha="right", va="top")
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.text(np.max(mean), md + 1.96*sd +  offset, 'Mean + 1.96*SD',ha="right", va="bottom")
    plt.text(np.max(mean), md + 1.96*sd - offset, f'{md + 1.96 * sd:.2f}', ha="right", va="top")
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.text(np.max(mean), md - 1.96*sd - offset, 'Mean - 1.96*SD', ha="right", va="top")
    plt.text(np.max(mean), md - 1.96 * sd + offset, f'{md - 1.96 * sd:.2f}', ha="right", va="bottom")
def writefile(filename,data):
    """
    函数说明：
    在指定地点创建一个文件，写入data数据
    :param filename:                需写入的文件地址和名称
    :param data:                    需写入的数据
    :return:                        无返回
    """
    data = data.astype('str')            #将data数组的数据转换成字符串
    with open(filename,'a+') as f :
        for str in data :
            f.write(str+'\n')
    return 0

def windows(data, num):
    """
    函数说明：
    对输入信号data分窗
    :param data:                  输入数据组
    :param num:                   输入规定每个窗的数据量
    :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
    """
    list = num                     #每个窗num个点
    row = int( len(data)/list)
    returndata = np.zeros((row,list))
    for i in range(row):
        for j in range(list):
            returndata[i][j] = data[i*num+j]
    return returndata

def windows_10(data, num):
    """
    函数说明：
    对输入信号data分窗
    :param data:                  输入数据组
    :param num:                   输入规定每个窗的数据量
    :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
    """
    list = num                     #每个窗num个点
    row = int( len(data)/list)
    returndata = []
    for i in range(row):
        if i==0 :
            returndata.append( data[0: 10000] )
        else:
            returndata.append( data[num*i-1000:num*(i+1)+1000])
    return returndata

# def windows_30(data, num):
#     """
#     函数说明：
#     对输入信号data分窗
#     :param data:                  输入数据组
#     :param num:                   输入规定每个窗的数据量
#     :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
#     """
#     list = num                     #每个窗num个点
#     row = int( len(data)/list)
#     returndata = []
#     for i in range(row):
#         if i==0 :
#             returndata.append( data[0: num] )
#         else:
#             returndata.append( data[num*i-5000:num*(i+1)+5000])
#     return returndata
def windows_30(data, state,num):
    """
    函数说明：
    对输入信号data分窗
    :param data:                  输入数据组
    :param num:                   输入规定每个窗的数据量
    :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
    """
    list = num                     #每个窗num个点
    row = int( len(data)/list)
    returndata = []
    State = []
    for i in range(row):
        if i==0 :
            returndata.append( data[0: num] )
            State.append(state[0:15])
        else:
            returndata.append( data[num*i-6000:num*(i+1)+6000])
            State.append(state[12 +(i-1)*15: (i-1)*15+33])
    return returndata,State




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

def Preprocessing(data):
    """
    对输入信号进行预处理:
           1.低通滤波
           2.移除基线
    :param data:        输入信号数据
    :return:            预处理后的信号数据
    """
    returndata = Butterworth(np.array( data ),type="lowpass",lowcut=20,order=4)
    baseline = signal.medfilt(returndata,351)
    returndata = returndata - baseline
    returndata = Butterworth(np.array(returndata), type="bandpass", lowcut=2.5,highcut=8.5, order=4)
    baseline = Butterworth(np.array(baseline), type="bandpass", lowcut=0.01,highcut=0.6, order=2)
    return returndata,baseline

def Dilate(x,N,g,M):
    """
    函数说明：
    对输入信号进行膨胀运算
    :param x:                     信号数据
    :param N:                     信号长度N
    :param g:                     结构信号
    :param M:                     结构信号长度M
    :return:
    """
    returndata = np.array([])
    for num in range(N-M+1):
        returndata = np.append(returndata,np.min( np.array(x[num:num+M])-np.array(g)) )
    return returndata

def Eorde(x,N,g,M):
    """
    函数说明：
    对输入信号进行腐蚀运算
    :param x:                     信号数据
    :param N:                     信号长度N
    :param g:                     结构信号
    :param M:                     结构信号长度M
    :return:
    """
    returndata = np.array([])
    for num in range(N-M+1):
        returndata = np.append(returndata,np.max( np.array(x[num:num+M])-np.array(g)) )
    return returndata

def Preprocessing2(data):
    """
    对输入信号进行预处理:
           1.低通滤波
           2.移除基线(形态滤波)
    :param data:        输入信号数据
    :return:            预处理后的信号数据
    """
    data = Butterworth(np.array( data ),type="lowpass",lowcut=20,order=4)
    #结构元宽度M，论文选择为采样频率的18%
    M = 200
    g = np.ones(200)
    Data_pre = np.insert(data, 0, np.zeros(200))
    Data_pre = np.insert(Data_pre, -1, np.zeros(200))
    #开运算:腐蚀+膨胀
    out1 = Eorde(Data_pre, len(Data_pre), g, M)     #腐蚀
    out2 = Dilate(out1, len(out1), g, M)    #膨胀
    out2 = np.insert(out2, 0, np.zeros(198))
    #闭运算:膨胀+腐蚀+腐蚀+膨胀
    out5 = Dilate(Data_pre, len(Data_pre), g, M)    #膨胀
    out6 = Eorde(out5, len(out5), g, M)     #腐蚀
    out6 = np.insert(out6, 0, np.zeros(198))

    baseline = (out2 + out6) / 2

    #-------------------------保留剩余价值------------------------
    returndata = Data_pre[:len(baseline)] - baseline
    returndata = np.delete(returndata, range(0,200), axis=0)
    returndata = returndata[:len(data)]
    baseline = baseline[200:]
    returndata[-1] = returndata[-2] = returndata[-3]
    baseline[-1] = baseline[-2] = baseline[-3]
    #-----------------------------------------------------------

    baseline = Butterworth(baseline, type="bandpass", lowcut=0.01, highcut=0.7, order=2)

    returndata = Butterworth(np.array(returndata), type="bandpass", lowcut=2, highcut=8.5, order=4)

    #plt.figure()
    #plt.subplot(1, 1, 1)
    #plt.plot(range(len(returndata)), returndata, 'green')
    #plt.plot(range(len(baseline)), baseline)
    #plt.show()

    return returndata,baseline

def Statedetect(data,threshold):
    """
    函数说明：
    将输入生理信号进行处理，移除大体动以及空床状态，只保留正常睡眠
    :param data:                输入信号数据
    :param threshold:           设置空床门槛
    :return:                    返还剩余的正常睡眠信号
    """
    win = windows(data, 2000)
    SD = np.zeros( win.shape[0] )
    Mean = np.zeros(win.shape[0])
    state = []

    for i in range( win.shape[0]):
        SD[i] = np.std(np.array(win[i]), ddof=1)
        Mean[i] = np.mean(np.array(abs(win[i])))
    Median_SD = np.median(SD)
    Median_Mean = np.median(Mean)

    for i in range( len(SD) ) :

        if SD[i] > (Median_SD*1.5) or  Mean[i] > (Median_Mean) + 50 or Mean[i] < (Median_Mean -50):
            state.append("Movement")
        elif SD[i] < threshold :
            state.append("Movement")
        else :
            state.append("Sleep")
    print('state:', state)

    new_state = copy.deepcopy(state)
    for i in range(len(state) -1 ):  #将体动前后2s的窗口都设置为体动
        if state[i] == "Movement":
            if i == 1:  #如果第一个窗口就是体动，则只将后一个2s置为体动
                new_state[i+1] = "Movement"
            else:
                new_state[i - 1] = "Movement"
                new_state[i + 1] = "Movement"
        else:
            pass
    print('new_state:', new_state)


    return  np.array( new_state )

def Artifacts_win(data):
    Max_value_set = []
    state = []
    win = windows(data, 2000)
    for i in range(len(win)):
        Max_value = np.max(win[i])
        Max_value_set.append(Max_value)
    len_30s = int(len(data) / 30000)
    len_60s = int(len(data) / 60000)
    len_120s = int(len(data) / 120000)
    len_300s = int(len(data) / 300000)
    i = 0
    while True:
        i_Count = 0
        if i +150 < len(Max_value_set):
            Max_value_30s = Max_value_set[i:i+15]   #提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i+30]   #提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:i + 60]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_300s = Max_value_set[i:i + 150]  # 提取300s内的最大值（150个2s的最大值）

        elif i + 150 > len(Max_value_set) and i + 60 < len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i + 30]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:i + 60]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_300s = Max_value_set[i:-1]  # 提取300s内的最大值（150个2s的最大值）
        elif i + 150 > len(Max_value_set) and i + 30 < len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i + 30]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:-1]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_300s = Max_value_set[i:-1]  # 提取300s内的最大值（150个2s的最大值）
        elif i + 150 > len(Max_value_set) and i + 15 < len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:-1]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:-1]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_300s = Max_value_set[i:-1]  # 提取300s内的最大值（150个2s的最大值）
        else:
            Max_value_30s = Max_value_set[i:-1]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:-1]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:-1]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_300s = Max_value_set[i:-1]  # 提取300s内的最大值（150个2s的最大值）
        Quartile_30s = np.percentile(Max_value_30s, 25)
        Quartile_60s = np.percentile(Max_value_60s, 25)
        Quartile_120s = np.percentile(Max_value_120s, 25)
        Quartile_300s = np.percentile(Max_value_300s, 25)
        Q1_30s = Quartile_30s*1.8#提取30s内最大值的四分位数的1.5倍作为基线
        Q1_60s = Quartile_60s * 2   # 提取60s内最大值的四分位数的1.5倍作为基线
        Q1_120s = Quartile_120s * 2  # 提取120s内最大值的四分位数的1.5倍作为基线
        Q1_300s = Quartile_300s * 2  # 提取300s内最大值的四分位数的1.5倍作为基线
        # print('Q1_30s',Q1_30s)
        # print('Q1_60s', Q1_60s)
        # print('Q1_120s', Q1_120s)
        # print('Q1_300s', Q1_300s)
        if Max_value_set[i] > Q1_30s or Max_value_set[i] < 0.4 *Quartile_30s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_60s or Max_value_set[i] < 0.4 *Quartile_60s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_120s or Max_value_set[i] < 0.4 *Quartile_120s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_300s or Max_value_set[i] < 0.4 *Quartile_300s:
            i_Count = i_Count + 1

        if i_Count > 1 :
            state.append("Movement")
        else:
            state.append("Sleep")

        i = i + 1

        if i >len(Max_value_set)-4:
            break

    new_state = copy.deepcopy(state)
    for i in range(len(state) - 1):  # 将体动前后2s的窗口都设置为体动
        if state[i] == "Movement":
            if i == 1:  # 如果第一个窗口就是体动，则只将后一个2s置为体动
                new_state[i + 1] = "Movement"
            else:
                new_state[i - 1] = "Movement"
                new_state[i + 1] = "Movement"
        else:
            pass
    # print('new_state:', new_state)
    Count_index = 0
    count = 0
    i = 0
    start_matrix = []
    end_matrix = []
    count_matrix = []
    while True:
        # print('i:',i)
        if i > len(new_state) - 1:
            break
        if new_state[i] == "Sleep" and Count_index == 0 and count == 0:
            i = i + 1
            pass
        elif new_state[i] == "Movement" and Count_index == 0 and count == 0:
            Count_index = 1
            start_index = i
            i = i + 1
        elif new_state[i] == "Movement" and Count_index == 1 and count == 0:
            Count_index = 1
            start_index = i
            i = i + 1
        elif new_state[i] == "Sleep" and Count_index == 1:
            count = count + 1
            i = i + 1
        elif new_state[i] == "Movement" and Count_index == 1 and count != 0:
            Count_index = 0
            end_index = i
            start_matrix.append(start_index)
            end_matrix.append(end_index)
            count_matrix.append(count)
            count = 0
            i = i + 1
    for i in range(len(start_matrix)):
        if 0 < count_matrix[i] < 5:
            list = ["Movement" for x in range(end_matrix[i] - start_matrix[i])]
            new_state[start_matrix[i]:end_matrix[i]] = list
    return  np.array(new_state)
def CutData(data,state):
    """
    函数说明：根据数组state的不同状态，对数据进行划分
    :param data:                        输入信号数组
    :param state:                       输入状态数组
    :return:                            返还切割后信号数组
    """
    Movestate = np.argwhere(state == "Movement").flatten().astype(int)
    cutdata = []
    mark = []
    count = 0
    if len(Movestate)==0 :
        mark.append( 0 )
        cutdata.append(data)
        return cutdata,mark
    for num in Movestate:
        if count != num * 2000 :
            cutdata.append( data[count:num*2000] )
            mark.append(count)
        else:
            pass
        count = (num+1)*2000
    if Movestate[-1] == int(len(data)/2000-1) :
        pass
    else:
        cutdata.append( data[(Movestate[-1]+1)*2000: ] )
        mark.append( (Movestate[-1]+1)*2000 )
    return cutdata,mark

def Shorttime_Energy(data):
    """
    :param data: 输入信号
    :return: 返回峰值平方能量和
    """
    data = data
    peaks = findpeak(data)
    roughs = findtrough(data)
    power_peaks = np.power(data[peaks],2)
    power_roughs = np.power(data[roughs],2)
    short_time_energy = np.sum(power_peaks) + np.sum(power_roughs)
    return short_time_energy



def InitBeatDetect(data,style="peak"):
    """
    函数说明：
    查找合理心跳点
    :param data:                  输入数据信号
    :param maxi_index:            输入数据峰值坐标
    :return:                      处理后的峰值坐标
    """
    length = len(data)
    # 创建新的索引和返回信号
    index = []
    # 创建当前峰和识别窗
    win_min = 0
    win_max = 1000
    while (True):
        if style=="peak":
            beat = int(np.argmax( data[win_min:win_max] ) + win_min)
        else:
            beat = int(np.argmin( data[win_min:win_max] ) + win_min)
        index.append(beat)
        win_min = max(0, beat + 500)
        win_max = min(length, beat + 1500)
        if (win_min >= length):
            break
    return np.array(index)

def distEuclidean(veca,vecb):
    """
    计算欧几里得距离
    """
    return np.sqrt( np.sum( np.square(veca-vecb) ) )


def normalize(data):
    """
    :param data:  输入信号
    :return:  返回归一化信号
    """
    data = data
    nor_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return nor_data
def ASD(veca,vecb):
    """
    计算绝对值的误差
    """
    return abs(np.sum(veca-vecb))

def SAD(veca,vecb):
    """
    计算误差绝对值的和
    """
    return np.sum(abs(veca-vecb))

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
            test.append(data[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))])  #N个0.7s的片段
    meanBCG = np.zeros(ModelLength)        # ----------------------对初始预判J峰的信号段相加平均
    for num in range(len(test)):
        meanBCG += test[num]
    meanBCG = meanBCG / len(test)
    dit = np.array([])                     # ----------------------计算初始预判信号与平均信号的相似性
    for num in range(len(test)):
        #para = 2 - ASD(test[num], meanBCG) / SAD(test[num], meanBCG)
        dit = np.append(dit, distEuclidean(test[num], meanBCG) * 1)

    indexmin = np.array([])                # -----------------------选择与平均信号最相似的1个原始信号
    for num in range(5):
        if len(dit)>1 :
            indexmin = np.append( indexmin, np.argmin(dit) )  #保存距离最小的5个0.7s片段的索引
            dit[np.argmin(dit)] = float("inf")
        else:
            pass
    indexmin = indexmin.astype(int)
    Model = np.zeros(ModelLength)

    for num in indexmin:
        Model += test[num]
    Model = Model/5

    #--------------------------------------
    chooseJ = np.full(len(data), np.nan)
    for num in np.array(Jpeak).astype(int):
        chooseJ[num] = data[num]       #提取初始J峰的值

    chooseModel = []
    for num in indexmin:
        chooseModel.append(test[num])   #提取距离最小的5个0.7s片段

    ChooseBCG = np.full(len(data), np.nan)
    for num in indexmin:
        ChooseBCG[int(Jpeak[num])-350:int(Jpeak[num])+350] = data[int(Jpeak[num])-350:int(Jpeak[num])+350]
    #--------------------------------------
    #plt.figure(figsize=(10, 4))
    #ax = plt.subplot(1, 1, 1)
    #plt.plot(np.linspace(-1, len(ECG) / 1000-1, len(ECG)), ECG/2, 'darkgrey', label='ECG')
    #plt.plot(np.linspace(-1, len(data) / 1000-1, len(data)), data*3.3/4096,'green',label='BCG')
    #plt.plot(np.linspace(-1, len(ChooseBCG) / 1000-1, len(ChooseBCG)), ChooseBCG*3.3/4096,'blue',label='Selected heartbeat')
    #plt.plot(np.linspace(-1, len(chooseJ) / 1000-1, len(chooseJ)), chooseJ*3.3/4096,'r.',markersize = 12,label='Initial estimate J-peak')
    #plt.tick_params(labelsize=12)
    #font2 = {'family': 'Times New Roman',
    #         'weight': 'normal',
    #         'size': 16,
    #         }
    #plt.xlabel('Time (s)', font2)
    #plt.ylabel('Amplitude (Volt)', font2)
    #plt.xticks([x for x in range(14)])
    #plt.legend(fontsize=10)
    #plt.show()
    return Model

def findpeak(data):
    """
    :param data:                  输入序列信号
    :return:                      返还峰峰值对应的坐标数组( 峰峰值相隔>1s )
    """
    #建立峰峰值数组和其对应的坐标
    maxi = np.zeros(len(data) - 2)
    maxi_index = []
    # 获取峰峰值点对应的x坐标
    for i in range(1, len(data) - 2):
        maxi[i] = np.where([(data[i] - data[i - 1] > 0) & (data[i] - data[i + 1] > 0)],data[i], np.nan)
        if np.isnan(maxi[i]):
            continue
        maxi_index.append(i)
    return np.array( maxi_index )

def findtrough(data):
    """
    函数说明：
    查找出输入信号的峰值点
    :param data:                  输入序列信号
    :return:                      返还峰值信号，非峰值点的信号段为np.nan
    """
    #建立峰峰值数组和其对应的坐标
    mini = np.zeros(len(data) - 2)
    mini_index = []
    # 获取峰峰值点对应的x坐标
    for i in range(1, len(data) - 2):
        mini[i] = np.where([(data[i] - data[i - 1] < 0) & (data[i] - data[i + 1] < 0)],data[i], np.nan)
        if np.isnan(mini[i]):
            continue
        mini_index.append(i)
    return np.array( mini_index )

def BeatDetection(data,envelop,MeanRR,up=1.86,down=0.53,style="peak"):
    """
    前向检测,根据Style选择合适的心跳位置
    :param data:                   输入数据信息
    :param up:                     上限倍数 Default = 1.86
    :param down:                   下限倍数 Default = 0.53
    :param style:                  根据峰或谷
    :return:                       心跳位置
    """
    length = len(data)
    envelop = np.array(envelop)
    data = np.array(data)
    # 心跳位置的索引
    index = []
    # 设置初始窗口
    win_min = 0
    win_max = 1500
    if style == "peak" :
        while( True ):
            peak = findpeak(data[win_min:win_max])
            peak = peak + win_min
            peak = peak.astype(int)
            if len(peak) == 0:
                break
            peakmax = np.argmax(data[peak])
            beat = peak[peakmax]
            if len(index) < 1 :               #首个检测
                if (beat>MeanRR*1.2) and len(peak)>1 :          #间隔过大，查询包络是否有峰值
                    EnvelopPeak = findpeak(envelop[win_min:beat]) + win_min
                    if len( EnvelopPeak ) == 0 :
                        index.append(beat)
                    else:
                        peak = np.array( [x for x in peak if x <(EnvelopPeak[-1]+350)] )
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        index.append(beat)
                else:
                    index.append(beat)
            else:
                dit = beat - index[-1]
                if len(index)==1 :
                    std = MeanRR
                else:
                    std = np.mean( np.diff(index) )
                    std = (std + MeanRR)/2
                while (dit > 1200) or (dit < 700) and len(peak)>1 :
                    if dit >1200 :
                        EnvelopPeak = findpeak(envelop[index[-1]:beat]) + index[-1]
                        EnvelopPeak = np.array( [x for x in EnvelopPeak if x > win_min+20] )
                        if len( EnvelopPeak ) == 0 :
                            if dit > up*std or dit < down*std :
                                peak = np.delete(peak, peakmax)
                                peakmax = np.argmax(data[peak])
                                beat = peak[peakmax]
                                dit = beat - index[-1]
                            else:
                                break
                        else:
                            peak = np.delete(peak, peakmax)
                            peak = np.array([x for x in peak if x < (EnvelopPeak[-1] + 350)])
                            peak = peak.astype(int)

                            if len(peak)==0 :
                                break
                            else:
                                pass

                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            if data[Senbeat] > data[beat] * 0.6:
                                beat = Senbeat
                            else:
                                break
                            dit = beat - index[-1]
                    elif dit > up*std or dit < down*std :
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        dit = beat - index[-1]
                    else:
                        break

                index.append(beat)
            win_min = max(0, index[-1] + 500)
            win_max = min(length, index[-1] + 1500)
            if (win_min > length-3) :
                break
    else:
        print("Vally is not exist!")
        while (True):
            peak = findtrough(data[win_min:win_max])
            peak = peak + win_min
            peak = peak.astype(int)
            if len(peak) == 0:
                break
            peakmax = np.argmax(data[peak])
            beat = peak[peakmax]
            if len(index) < 2:  # 首个检测
                if (beat > MeanRR * 1.2) and len(peak) > 1:  # 间隔过大，查询包络是否有峰值
                    EnvelopPeak = findpeak(envelop[win_min:beat])
                    if len(EnvelopPeak) == 0:
                        index.append(beat)
                    else:
                        peak = np.where(peak < (EnvelopPeak[0] + 300))
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        index.append(beat)
                else:
                    index.append(beat)
            else:
                dit = beat - index[-1]
                std = np.mean(np.diff(index))
                std = (std + MeanRR) / 2
                while (dit > std * up) or (dit < std * down) and len(peak) > 1:
                    EnvelopPeak = findpeak(envelop[win_min:beat])
                    if len(EnvelopPeak) == 0:
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        dit = beat - index[-1]
                    else:

                        peak = np.where(peak < (EnvelopPeak[0] + 300))
                        peakmax = np.argmax(data[peak])
                        Senbeat = peak[peakmax]
                        if data[Senbeat] > data[beat]*0.4 :
                            beat = Senbeat
                        else:
                            beat = beat
                        dit = beat - index[-1]

                index.append(beat)
            win_min = max(0, index[-1] + 500)
            win_max = min(length, index[-1] + 1500)
            if (win_min > length - 3):
                break
    return index

def BeatChoose(BCGcor,BCGdit,cor_f, cor_b, dit_f, dit_b, initInterval):
    """
    函数解释:根据Cor和Dit曲线检测的心跳位置，来定位最终心跳位置
    :param BCGcor:               相关信号
    :param BCGdit:               形态距离信号
    :param cor_f:               相关前向检测心跳位置
    :param cor_b:               相关后向检测心跳位置
    :param dit_f:               距离前向检测心跳位置
    :param dit_b:               距离后向检测心跳位置
    :param BCGCor:              BCG相关曲线
    :return:                    最终确定的心跳位置
    """
    BCGcor = BCGcor
    BCGdit = BCGdit
    BeatPosition = np.array([])
    num0,num1,num2,num3 = 0,0,0,0
    while(True):
        if num0>=len(cor_f) or num1>=len(cor_b) or num2>=len(dit_f) or num3>=len(dit_b) :
            break
        else:   #提取相关前向/后向点、形态距前向/后向点
            beat = np.array([
                cor_f[num0],cor_b[num1],
                dit_f[num2],dit_b[num3]
            ])

            # 移除间隔小于500的点
            beat_detect = np.array([0, 1, 2, 3])
            if len(BeatPosition)>0 :
                beat_detect = np.where( (beat - BeatPosition[-1])<500 )[0]
                if len(beat_detect)==0 :
                    pass
                else:
                    if 0 in beat_detect:
                        num0 = num0 + 1
                    if 1 in beat_detect:
                        num1 = num1 + 1
                    if 2 in beat_detect:
                        num2 = num2 + 1
                    if 3 in beat_detect:
                        num3 = num3 + 1
                    continue
            else:
                pass

            # 找出最小位置
            if len(BeatPosition) > 2: #若检测到三个点后
                initInterval = (initInterval + np.mean(np.diff(BeatPosition))) // 2    #间期设置为根据之前检测到的点求平均值

            Minibeat = np.min(beat)  #取四个点中最小的值
            beat_choose = np.where((beat - Minibeat) <= (initInterval // 2))[0]


            case1 = False           #cor b=f    #case1 = true表示相关信号定位正确(相关信号前向和后向定位的差值小于30ms则判为正确)
            case2 = False           #dit b=f    #case2 = true表示形态距信号定位正确(形态距信号前向和后向定位的差值小于30ms则判为正确)

            if 1 and 0 in beat_choose:
                if abs(beat[0] - beat[1]) < 30 :    # 相关信号前向和后向定位的差值小于30ms则判为正确
                    case1 = True
                else:
                    case1 = False    # 相关信号前向和后向定位的差值大于30ms则判为正确
            else:
                case1 = False

            if 2 and 3 in beat_choose:
                if abs(beat[2] - beat[3]) < 30 :  #形态距信号前向和后向定位的差值小于30ms则判为正确
                    case2 = True
                else:
                    case2 = False      #形态距信号前向和后向定位的差值大于30ms则判为正确
            else:
                case2 = False


            beat = np.array(beat[beat_choose])

            if len(beat)>0 :
                if case2 and case1 :
                    if abs(beat[0] - beat[2]) >200: #如果形态距和相关都检测正确但索引指向不同峰，则取相关作为最终定位点
                        pos = beat[0]
                    else:#如果形态距和相关都检测正确且索引相同，则取两者的均值作为最终定位点
                        pos = np.mean(beat)
                elif case1 :    #如果只有相关检测正确，则将相关前向检测点作为最终点位点
                    pos = beat[0]
                elif case2 :
                    pos = beat[-1]  #如果只有形态距检测正确，则将形态后向检测点作为最终点位点
                else:
                    beat = beat.astype(int)
                    if len(beat)==1 :                        #长度为1时，取该点BCGCor和前3个心跳的BCGCor相比较,再和前面的RR间期相比较
                        pos = beat[0]
                    else :
                        if len(beat) == 2:
                             pos = np.mean(beat)   #若只有两个点，则取平均作预估心跳点
                        elif len(beat) == 3: #如有三个点，则两两相减取最小
                            sub_distance = [abs(beat[0]-beat[1]),abs(beat[0]-beat[2]),abs(beat[1]-beat[2])]
                            mean_distance = np.mean(sub_distance)
                            if mean_distance > 150:#如果误差太大，则选择最接近均值的那个点作为输出
                                mean_beat = np.mean(beat)
                                pos = beat[np.argmin(abs(beat - mean_beat))]
                            else:
                                min_dist_index = np.argmin(sub_distance)
                                if min_dist_index == 0:
                                    pos_indics = [beat[0],beat[1]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 1:
                                    pos_indics = [beat[0],beat[2]]
                                    pos= np.mean(pos_indics)
                                else:
                                    pos_indics = [beat[1],beat[2]]
                                    pos= np.mean(pos_indics)
                        else: #如果有四个点，也是两两相减取最小
                            sub_value = [abs(BCGcor[beat[0]] - BCGdit[beat[2]]), abs(BCGcor[beat[0]] - BCGdit[beat[3]]), abs(BCGcor[beat[1]] - BCGdit[beat[2]]), abs(BCGcor[beat[1]] - BCGdit[beat[3]])]
                            sub_distance = [abs(beat[0] - beat[1]),
                                            abs(beat[0] - beat[2]),
                                            abs(beat[0] - beat[3]),
                                            abs(beat[1] - beat[2]),
                                            abs(beat[1] - beat[3]),
                                            abs(beat[2] - beat[3])]
                            # sub_distance = [abs(beat[0] - beat[2]),
                            #                 abs(beat[0] - beat[3]),
                            #                 abs(beat[1] - beat[2]),
                            #                 abs(beat[1] - beat[3])]
                            mean_distance = np.mean(sub_distance)
                            if mean_distance > 100 : #如果索引差距太大，则选择峰值最接近的作为最终输出
                                sub_value_index = np.argmin(sub_value)
                                if sub_value_index == 0: #如果前向相关和前向形态距检测的幅值接近，则将其索引平均值作输出
                                    pos_indics = [beat[0], beat[2]]
                                    pos = np.mean(pos_indics)
                                elif sub_value_index == 1: #如果前向相关和后向形态距检测的幅值接近，则将其索引平均值作输出
                                     pos_indics = [beat[0],beat[3]]
                                     pos= np.mean(pos_indics)
                                elif sub_value_index == 2: #如果后向相关和前向形态距检测的幅值接近，则将其索引平均值作输出
                                    pos_indics = [beat[1], beat[2]]
                                    pos = np.mean(pos_indics)
                                else:
                                    pos_indics = [beat[1], beat[3]]
                                    pos = np.mean(pos_indics)
                            else:
                                min_dist_index = np.argmin(sub_distance)
                                if min_dist_index == 0:
                                    pos_indics = [beat[0],beat[1]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 1:
                                    pos_indics = [beat[0],beat[2]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 2:
                                    pos_indics = [beat[0],beat[3]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 3:
                                    pos_indics = [beat[1],beat[2]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 4:
                                    pos_indics = [beat[1],beat[3]]
                                    pos= np.mean(pos_indics)
                                else:
                                    pos_indics = [beat[2],beat[3]]
                                    pos= np.mean(pos_indics)

                #----------判断间期是否接受
                if pos==0 :
                    pass
                elif len(BeatPosition)==1 :
                    if 500<pos-BeatPosition[-1]<2000:
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                elif len(BeatPosition)>1 :
                    Interval = BeatPosition[-1] - BeatPosition[-2]   #取上两个点求间期
                    if 500<pos-BeatPosition[-1]<2000:
                        BeatPosition = np.append(BeatPosition, pos)
                    elif (pos-BeatPosition[-1]>2001):
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                # 首个不用判断
                else:
                    BeatPosition = np.append(BeatPosition, pos)
            else:
                pass
            #----------num+1

            if 0 in beat_choose or 0 in beat_detect :
                num0 = num0 + 1
            if 1 in beat_choose or 1 in beat_detect :
                num1 = num1 + 1
            if 2 in beat_choose or 2 in beat_detect :
                num2 = num2 + 1
            if 3 in beat_choose or 3 in beat_detect :
                num3 = num3 + 1
    return BeatPosition

def BeatChoose1(cor_f, cor_b, dit_f, dit_b, initInterval):
    """
    函数解释:根据Cor和Dit曲线检测的心跳位置，来定位最终心跳位置
    :param cor_f:               相关前向检测心跳位置
    :param cor_b:               相关后向检测心跳位置
    :param dit_f:               距离前向检测心跳位置
    :param dit_b:               距离后向检测心跳位置
    :param BCGCor:              BCG相关曲线
    :return:                    最终确定的心跳位置
    """
    BeatPosition = np.array([])
    num0,num1,num2,num3 = 0,0,0,0
    while(True):
        if num0>=len(cor_f) or num1>=len(cor_b) or num2>=len(dit_f) or num3>=len(dit_b) :
            break
        else:   #提取相关前向/后向点、形态距前向/后向点
            beat = np.array([
                cor_f[num0],cor_b[num1],
                dit_f[num2],dit_b[num3]
            ])

            # 移除间隔小于500的点
            beat_detect = np.array([0, 1, 2, 3])
            if len(BeatPosition)>0 :
                beat_detect = np.where( (beat - BeatPosition[-1])<500 )[0]
                if len(beat_detect)==0 :
                    pass
                else:
                    if 0 in beat_detect:
                        num0 = num0 + 1
                    if 1 in beat_detect:
                        num1 = num1 + 1
                    if 2 in beat_detect:
                        num2 = num2 + 1
                    if 3 in beat_detect:
                        num3 = num3 + 1
                    continue
            else:
                pass

            # 找出最小位置
            if len(BeatPosition) > 2: #若检测到三个点后
                initInterval = (initInterval + np.mean(np.diff(BeatPosition))) // 2    #间期设置为根据之前检测到的点求平均值

            Minibeat = np.min(beat)  #取四个点中最小的值
            beat_choose = np.where((beat - Minibeat) <= (initInterval // 2))[0]


            case1 = False           #cor b=f    #case1 = true表示相关信号定位正确(相关信号前向和后向定位的差值小于30ms则判为正确)
            case2 = False           #dit b=f    #case2 = true表示形态距信号定位正确(形态距信号前向和后向定位的差值小于30ms则判为正确)

            if 1 and 0 in beat_choose:
                if abs(beat[0] - beat[1]) < 30 :    # 相关信号前向和后向定位的差值小于30ms则判为正确
                    case1 = True
                else:
                    case1 = False    # 相关信号前向和后向定位的差值大于30ms则判为正确
            else:
                case1 = False

            if 2 and 3 in beat_choose:
                if abs(beat[2] - beat[3]) < 30 :  #形态距信号前向和后向定位的差值小于30ms则判为正确
                    case2 = True
                else:
                    case2 = False      #形态距信号前向和后向定位的差值大于30ms则判为正确
            else:
                case2 = False


            beat = np.array(beat[beat_choose])

            if len(beat)>0 :
                if case2 and case1 :     #如果形态距和相关都检测正确，则取两者的均值作为最终定位点
                    pos = np.mean(beat)
                elif case1 :    #如果只有相关检测正确，则将相关前向检测点作为最终点位点
                    pos = beat[0]
                elif case2 :
                    pos = beat[-1]  #如果只有形态距检测正确，则将形态后向检测点作为最终点位点
                else:
                    beat = beat.astype(int)
                    if len(beat)==1 :                        #长度为1时，取该点BCGCor和前3个心跳的BCGCor相比较,再和前面的RR间期相比较
                        pos = beat[0]
                    else :
                        if len(beat) == 2:
                             pos = np.mean(beat)   #若只有两个点，则取平均作预估心跳点
                        elif len(beat) == 3: #如有三个点，则两两相减取最小
                            sub_distance = [abs(beat[0]-beat[1]),abs(beat[0]-beat[2]),abs(beat[1]-beat[2])]
                            mean_distance = np.mean(sub_distance)
                            if mean_distance > 150:#如果误差太大，则选择最接近均值的那个点作为输出
                                mean_beat = np.mean(beat)
                                pos = beat[np.argmin(abs(beat - mean_beat))]
                            else:
                                min_dist_index = np.argmin(sub_distance)
                                if min_dist_index == 0:
                                    pos_indics = [beat[0],beat[1]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 1:
                                    pos_indics = [beat[0],beat[2]]
                                    pos= np.mean(pos_indics)
                                else:
                                    pos_indics = [beat[1],beat[2]]
                                    pos= np.mean(pos_indics)
                        else: #如果有四个点，也是两两相减取最小
                            sub_distance = [abs(beat[0] - beat[1]), abs(beat[0] - beat[2]), abs(beat[0] - beat[3]), abs(beat[1] - beat[2]), abs(beat[1] - beat[3]), abs(beat[2] - beat[3])]
                            mean_distance = np.mean(sub_distance)
                            if mean_distance > 150 : #如果误差太大，则选择最接近均值的那个点作为输出
                                mean_beat = np.mean(beat)
                                pos = beat[np.argmin(abs(beat - mean_beat))]
                            else:
                                min_dist_index = np.argmin(sub_distance)
                                if min_dist_index == 0:
                                    pos_indics = [beat[0],beat[1]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 1:
                                    pos_indics = [beat[0],beat[2]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 2:
                                    pos_indics = [beat[0],beat[3]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 3:
                                    pos_indics = [beat[1],beat[2]]
                                    pos= np.mean(pos_indics)
                                elif min_dist_index == 4:
                                    pos_indics = [beat[1],beat[3]]
                                    pos= np.mean(pos_indics)
                                else:
                                    pos_indics = [beat[2],beat[3]]
                                    pos= np.mean(pos_indics)

                #----------判断间期是否接受
                if pos==0 :
                    pass
                elif len(BeatPosition)==1 :
                    if 500<pos-BeatPosition[-1]<2000:
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                elif len(BeatPosition)>1 :
                    Interval = BeatPosition[-1] - BeatPosition[-2]   #取上两个点求间期
                    if 500<pos-BeatPosition[-1]<2000:
                        BeatPosition = np.append(BeatPosition, pos)
                    elif (pos-BeatPosition[-1]>2001):
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                # 首个不用判断
                else:
                    BeatPosition = np.append(BeatPosition, pos)
            else:
                pass
            #----------num+1

            if 0 in beat_choose or 0 in beat_detect :
                num0 = num0 + 1
            if 1 in beat_choose or 1 in beat_detect :
                num1 = num1 + 1
            if 2 in beat_choose or 2 in beat_detect :
                num2 = num2 + 1
            if 3 in beat_choose or 3 in beat_detect :
                num3 = num3 + 1
    return BeatPosition
def BeatDetection3(data, envelop):
    tempplate = TemplateMarching(data, samplate_rate=1000)
    beat = tempplate.beatDetection(data, envelop=envelop, maxs=1.4)
    return beat

def fineTun(data, peaks, th=300):
    return_peak = []
    raw_interval = np.diff(peaks)
    raw_interval = np.delete(raw_interval,np.where(raw_interval > 2000))#求除体动以外的间期均值

    mean_interval = np.mean(raw_interval)
    # print('平均间期为：',mean_interval)

    for peak in peaks :
        if int(peak) > len(data):
            break
        #求当前峰值和与它最近的K谷的JK差值
        min_win, max_win = max(0, int(peak - 50)), min(len(data), int(peak + th))
        new_peakindex = findpeak(data[min_win: max_win]) + min_win
        peak_indics = abs(new_peakindex - peak)
        peak1 = new_peakindex[np.argmin(peak_indics)]  # 确定当前峰索引
        new_roughindex = findtrough(data[min_win: max_win]) + min_win
        rough_indics = abs(new_roughindex - peak1)
        if len(rough_indics) == 0:
            break
        else:
            rough = new_roughindex[np.argmin(rough_indics)]  # 确定最近的谷索引
            InitJK = data[int(peak1)] - data[int(rough)]  # 求初始确定峰值的JK差值


        min_win, max_win = max(0, int(peak-th)), min(len(data), int(peak+th))
        new_peakindex = findpeak(data[min_win: max_win]) + min_win
        if int(peak) in new_peakindex or len(return_peak) ==0:  #如果定位点是峰值点，则继续检测下一点
            return_peak.append(peak)
            continue
        else: #如果定位点不是峰值点，则在该点的前后th ms内重新选一个满足间期条件的最大JK差的峰值作为新的定位点
            JKmin_win = new_peakindex[np.argmin(new_peakindex)] - 10
            JKmax_win = new_peakindex[np.argmax(new_peakindex)] + 150
            Amp_JK,index = Amp_JKDetection(data[JKmin_win: JKmax_win])
            if (np.mean(Amp_JK))< 15 or len(Amp_JK) ==0  : #如果检测到所有的JK值过小（即该范围没有符合的J峰候选），则加大检测范围
                JKmin_win = int(peak - 2*th)
                JKmax_win = int(peak + 2*th )
                Amp_JK, index = Amp_JKDetection(data[JKmin_win: JKmax_win])
            index = index + JKmin_win
            if len(Amp_JK) ==0:
                min_win, max_win = max(0, int(peak - 1.5*th)), min(len(data), int(peak + 1.5*th))
                new_peakindex = findpeak(data[min_win: max_win]) + min_win
                max_index = np.argmax(data[new_peakindex])
                new_peakindex = new_peakindex[max_index]
                return_peak.append(new_peakindex)
            else:
                max_JK_index = np.argmax(Amp_JK)
                new_index = index[max_JK_index]#返回最大JK值的索引
                dit = new_index - return_peak[-1]
                while (dit <500) or (dit > mean_interval*1.5)  :
                    if len(Amp_JK) == 1 :
                        new_index = np.argmax(data[min_win: max_win]) + min_win
                        # return_peak.append(new_index)
                        break
                    else:
                        Amp_JK = np.delete(Amp_JK,max_JK_index)
                        index = np.delete(index, max_JK_index)
                        max_JK_index = np.argmax(Amp_JK)
                        new_index = index[max_JK_index]  # 返回最大JK值的索引
                        dit = new_index - return_peak[-1]
                # if data[new_index] > data[int(peak)]*0.9 or Amp_JK[max_JK_index] >InitJK*1.5 :#如果新的定位点满足间期大于500ms且小于900*1.2ms，且新的峰值比原本峰值大则保存新点
                if Amp_JK[max_JK_index] > InitJK * 1.5:  # 如果新的定位点满足间期大于500ms且小于900*1.2ms，且新的峰值比原本峰值大则保存新点
                    return_peak.append(new_index)
                else:#否则，在原本点附近找一个最接近的峰值点
                    min_win, max_win = max(0, int(peak - th)), min(len(data), int(peak + th))
                    new_peakindex = findpeak(data[min_win: max_win]) + min_win
                    which_peaks = abs(new_peakindex - peak)
                    min_index = np.argmin(which_peaks)
                    choose_peak = new_peakindex[min_index]
                    return_peak.append(choose_peak)



    return np.array(return_peak)

def fineTun3(data, peaks, th=300):
    return_peak = np.array([])
    # return_peak = []
    raw_interval = np.diff(peaks)
    raw_interval = np.delete(raw_interval,np.where(raw_interval > 2000))#求除体动以外的间期均值
    mean_interval = np.mean(raw_interval)
    # print('平均间期为：',mean_interval)

    for peak in peaks :
        if int(peak) > len(data):
            break
        if len(return_peak) >3:
            if peak - return_peak[-1] < mean_interval*0.65:  #检测上一个误检点微调后，当前点是否由于上一个点误检而检测出错
                continue #如果当前点是由于上一个点的误检导致的，则直接跳过当前点，检测下一个点
        #求当前峰值和与它最近的K谷的JK差值
        min_win, max_win = max(0, int(peak - th)), min(len(data), int(peak + th))
        new_peakindex = findpeak(data[min_win: max_win]) + min_win
        peak_indics = abs(new_peakindex - peak)
        peak1 = new_peakindex[np.argmin(peak_indics)]  # 确定当前峰索引
        new_roughindex = findtrough(data[min_win: max_win]) + min_win
        rough_indics = abs(new_roughindex - peak1)
        if len(rough_indics) == 0:
            continue
        else:
            rough = new_roughindex[np.argmin(rough_indics)]  # 确定最近的谷索引
            InitJK = data[int(peak1)] - data[int(rough)]  # 求初始确定峰值的JK差值


        min_win, max_win = max(0, int(peak-th)), min(len(data), int(peak+th))
        new_peakindex = findpeak(data[min_win: max_win]) + min_win
        if int(peak) in new_peakindex or len(return_peak) ==0:  #如果定位点是峰值点，则继续检测下一点
            # return_peak.append(peak)
            return_peak = np.append(return_peak,peak)
            continue
        else: #如果定位点不是峰值点，则在该点的前后th ms内重新选一个满足间期条件的最大JK差的峰值作为新的定位点
            JKmin_win = new_peakindex[np.argmin(new_peakindex)] - 10
            JKmax_win = new_peakindex[np.argmax(new_peakindex)] + 200
            Amp_JK,index = Amp_JKDetection(data[JKmin_win: JKmax_win])
            if (np.mean(Amp_JK))< 15 or len(Amp_JK) ==0  : #如果检测到所有的JK值过小（即该范围没有符合的J峰候选），则加大检测范围
                JKmin_win = int(peak - 2*th)
                JKmax_win = int(peak + 2*th )
                Amp_JK, index = Amp_JKDetection(data[JKmin_win: JKmax_win])
            index = index + JKmin_win
            if return_peak[-1] in index:
                cut_location = np.where(index == return_peak[-1])
                index = np.delete(index,cut_location)
                Amp_JK = np.delete(Amp_JK,cut_location)
            if len(Amp_JK) ==0:
                min_win, max_win = max(0, int(peak - 1.5*th)), min(len(data), int(peak + 1.5*th))
                new_peakindex = findpeak(data[min_win: max_win]) + min_win
                max_index = np.argmax(data[new_peakindex])
                new_peakindex = new_peakindex[max_index]
                # return_peak.append(new_peakindex)
                return_peak = np.append(return_peak,new_index)
            else:
                max_JK_index = np.argmax(Amp_JK)
                new_index = index[max_JK_index]#返回最大JK值的索引
                dit = new_index - return_peak[-1]
                while (dit <500) or (dit > mean_interval*1.5)  :
                    if len(Amp_JK) == 1 :
                        new_index = np.argmax(data[min_win: max_win]) + min_win
                        # return_peak.append(new_index)
                        break
                    else:
                        cut_index = max_JK_index
                        Amp_JK = np.delete(Amp_JK,cut_index)
                        index = np.delete(index, cut_index)
                        max_JK_index = np.argmax(Amp_JK)
                        new_index = index[max_JK_index]  # 返回最大JK值的索引
                        dit = new_index - return_peak[-1]
                # if data[new_index] > data[int(peak)]*0.9 or Amp_JK[max_JK_index] >InitJK*1.5 :#如果新的定位点满足间期大于500ms且小于900*1.2ms，且新的峰值比原本峰值大则保存新点
                if Amp_JK[max_JK_index] > InitJK * 1.2 and data[new_index] > data[int(peak)]*0.9 :  # 如果新的定位点满足间期大于500ms且小于900*1.2ms，且新的峰值比原本峰值大则保存新点
                    # return_peak.append(new_index)
                    return_peak = np.append(return_peak, new_index)
                else:#否则，在原本点附近找一个最接近的峰值点
                    min_win, max_win = max(0, int(peak - th)), min(len(data), int(peak + th))
                    # new_peakindex = findpeak(data[min_win: max_win]) + min_win
                    # which_peaks = abs(new_peakindex - peak)
                    # min_index = np.argmin(which_peaks)
                    # choose_peak = new_peakindex[min_index]
                    # return_peak.append(choose_peak)
                    Amp_JK, index = Amp_JKDetection(data[min_win: max_win])
                    index = index + min_win
                    if len(Amp_JK) >0:
                        max_JK_index = np.argmax(Amp_JK)
                        new_index = index[max_JK_index]
                        return_peak = np.append(return_peak, new_index)
                    else:
                        return_peak = np.append(return_peak, peak1)

    return_peak = return_peak.astype(int)





    return return_peak

def fineTun2(data, peaks, th=300):
    return_peak =np.array([])
    for peak in peaks :
        if peak > len(data):
            break
        min_win, max_win = max(0, int(peak-th)), min(len(data), int(peak+th))
        new_peakindex = findpeak(data[min_win: max_win]) + min_win
        if peak in new_peakindex or len(return_peak) ==0:  #如果定位点是峰值点，则继续检测下一点
            return_peak = np.append(return_peak,peak)
            continue
        else: #如果定位点不是峰值点，则在该点的前后th ms内重新选一个满足间期条件的最大JK差的峰值作为新的定位点
            Amp_JK,index = Amp_JKDetection(data[min_win: max_win])
            index = index + min_win
            max_JK_index = np.argmax(Amp_JK)
            new_index = index[max_JK_index]#返回最大JK值的索引
            dit = new_index - return_peak[-1]
            while (dit <500) or (dit > 900*1.5)  :
                if len(Amp_JK) == 1:
                    new_index = np.argmax(data[min_win: max_win]) + min_win
                    return_peak = np.append(return_peak, new_index)
                    break
                else:
                    Amp_JK = np.delete(Amp_JK,max_JK_index)
                    max_JK_index = np.argmax(Amp_JK)
                    new_index = index[max_JK_index]  # 返回最大JK值的索引
                    dit = new_index - return_peak[-1]
            return_peak = np.append(return_peak, new_index)#如果新的定位点满足间期大于500ms且小于900*1.2ms，且新的峰值比原本峰值大则保存新点


    return return_peak

def InitalfineTun(data, peaks, th=10):
    return_peak = np.array([])
    for peak in peaks :
        if peak > len(data):continue
        min_win, max_win = max(0, int(peak-th)), min(len(data), int(peak+th))
        # new_peakindex = findpeak(data[min_win: max_win]) + min_win
        new_index = np.argmax(data[min_win: max_win]) + min_win
        # if new_index in new_peakindex:
        return_peak= np.append(return_peak,new_index)
    return return_peak



def Amp_JKDetection(data,style="peak"):
    """
    函数说明：
    查找合理心跳点
    :param data:                  输入数据信号
    :param maxi_index:            输入数据峰值坐标
    :return:                      处理后的峰值坐标
    """
    Amp_JK = np.zeros(len(data))
    BCG_Pindex = findpeak(data)
    BCG_Vindex = findtrough(data)

    for peak in BCG_Pindex:
        Minrange = max(0, peak)
        Maxrange = min(len(data), peak + 150)
        chooseV = [int(x) for x in BCG_Vindex if Minrange < x < Maxrange]
        if len(chooseV) == 0:
            continue
        chooseP = np.full(len(chooseV), data[peak])
        MaxAm = np.max(chooseP - data[chooseV])
        Amp_JK[int(peak)] = MaxAm
    index = np.where(Amp_JK != 0)[0]
    return np.array(Amp_JK[index]),index #返回片段内每个峰的JK的幅值差和对应的峰值索引
#定义全局参数
Sample_org  = 1000
Modellength = 700

if __name__ == '__main__':
    #打开文件
    # id = 89
    # orgBCG     = pd.read_csv("D:\David\dataset2021\ECG&BCG\671\BCG.txt"%(str(id))).to_numpy().reshape(-1)
    # locations_J = pd.read_csv("E:\Code\HRV_DETECTION2\dataset\%s\location_R.txt"%(str(id))).to_numpy().reshape(-1)
    txtPath = "/home/qz/文档/Qz/WorkSpace/ArtifactDataset/岑玉娴2018060622-16-25/raw_org.txt"
    txtName = os.path.basename(txtPath)[:-4] + 'Movement' + '.txt'
    orgBCG     = pd.read_csv(txtPath).to_numpy().reshape(-1)
    # locations_J = pd.read_csv("E:/David/ECG&BCG/03/res_Ecg&Bcg_1/new_R_peak.txt").to_numpy().reshape(-1)
    # locations_J = locations_J.astype(int)

    MovementLabel = []

    #ECG        = pd.read_csv("E:\Code\HRV_DETECTION2\dataset\%s\ECG.txt"%(str(id))).to_numpy().reshape(-1)
    start_point = 10800000
    # start_point = 0
    end_point = start_point + 3600000
    orgBCG = orgBCG[start_point:end_point]
    print("orgBCG长度为：",len(orgBCG))
    filter_bcg = Butterworth(orgBCG,type='bandpass',lowcut = 2,highcut = 15,order = 2)
    All_state = Artifacts_win(filter_bcg)
    # All_state = Statedetect(filter_bcg, 0.1)
    #----------------------显示信号时，体动信号标红，正常信号标绿--------------------------#
    Movement_data = np.full(len(orgBCG), np.nan)
    Normal_data = np.full(len(orgBCG), np.nan)
    plt.figure(figsize=(12,8))
    MovementWriteNum = 1    #体动序号
    for i in range(len(All_state)):
        if All_state[i] == "Movement":
            Movement_data[i * 2000:(i + 1) * 2000] = orgBCG[i * 2000:(i + 1) * 2000]
            # plt.scatter(i * 2000, Movement_data[i * 2000])
            if i != 0 and All_state[i-1] != "Movement":
                plt.vlines(i * 2000, 1775, 1875, color="blue")  # 竖线
                MovementLabel.append(MovementWriteNum)
                MovementLabel.append(i * 2000)
                MovementWriteNum += 1
            if All_state[i + 1] != "Movement":
                plt.vlines((i + 1) * 2000, 1775, 1875, color="blue")  # 竖线
                MovementLabel.append((i + 1) * 2000)

        else:
            Normal_data[i * 2000:(i + 1) * 2000] = orgBCG[i * 2000:(i + 1) * 2000]

    print(MovementLabel)
    with open(txtName, 'w', encoding='utf-8') as file:
        for i in range(len(MovementLabel)):
            file.write(str(MovementLabel[i]) + "\n")

    plt.plot( Movement_data,color = 'red', label="Movement_data")
    plt.plot( Normal_data, color='green', label="Normal_data")
    list_point = end_point - start_point
    # for i in range(len(All_state)):
    #     if All_state[i] == "Movement":
    #         plt.scatter(i * 2000, Movement_data[i * 2000])
    # x = np.arange(0,3600000)
    # plt.plot(x, Movement_data[All_state], label='list1', marker="o", markersize=1)  # marker设置标记形状 markersize设置标记大小
    plt.legend(ncol = 2)
    plt.show()


















