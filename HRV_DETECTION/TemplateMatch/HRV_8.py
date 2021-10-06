# _*_ coding: utf-8 _*_

"""
@ date:             2020-09-20
@ author:           jingxian
@ illustration:     Beat detection by template matching
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import copy
import datetime
from TemplateMatch import Investiage

from scipy import signal

from scipy import fftpack


def openfile(filename,num=0):
    """
    函数说明：
    打开txt文件,获取文件中的数据，以每行为一数据，将其放入数组filedata[]
    :param filename:                需打开txt的名称
    :param num:                     选择第几列
    :return:                        返还文件数据数组及长度
    """
    filedata = []
    with open(filename,'r',encoding='UTF-8') as f:
        for line in f.readlines() :
            c_array = line.strip().split('\t')
            filedata.append( float(c_array[num]) )
    return filedata

def openfile2(filename,num):
    """
    函数说明：
    打开txt文件,获取文件中的数据，以每行为一数据，将其放入数组filedata[]
    :param filename:                需打开txt的名称
    :param num:                     选择第几列
    :return:                        返还文件数据数组及长度
    """
    filedata = []
    with open(filename,'r',encoding='UTF-8') as f:
        for line in f.readlines() :
            c_array = line.strip().split('\t')
            filedata.append( float(c_array[num]) )
    return filedata

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
            returndata.append( data[0: 1000] )
        else:
            returndata.append( data[num*i-100:num*(i+1)+100])
    return returndata

def windows_30(data, num):
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
            returndata.append( data[0: num] )
        else:
            returndata.append( data[num*i-5000:num*(i+1)+5000])
    return returndata

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
    M = 20
    g = np.ones(20)
    Data_pre = np.insert(data, 0, np.zeros(20))
    Data_pre = np.insert(Data_pre, -1, np.zeros(20))
    #开运算:腐蚀+膨胀
    out1 = Eorde(Data_pre, len(Data_pre), g, M)     #腐蚀
    out2 = Dilate(out1, len(out1), g, M)    #膨胀
    out2 = np.insert(out2, 0, np.zeros(18))
    #闭运算:膨胀+腐蚀+腐蚀+膨胀
    out5 = Dilate(Data_pre, len(Data_pre), g, M)    #膨胀
    out6 = Eorde(out5, len(out5), g, M)     #腐蚀
    out6 = np.insert(out6, 0, np.zeros(18))

    baseline = (out2 + out6) / 2

    #-------------------------保留剩余价值------------------------
    returndata = Data_pre[:len(baseline)] - baseline
    returndata = np.delete(returndata, range(0,20), axis=0)
    returndata = returndata[:len(data)]
    baseline = baseline[20:]
    returndata[-1] = returndata[-2] = returndata[-3]
    baseline[-1] = baseline[-2] = baseline[-3]
    #-----------------------------------------------------------


    baseline = Butterworth(baseline, type="bandpass", lowcut=0.01, highcut=0.7, order=2)

    #returndata = Butterworth(np.array(returndata), type="bandpass", lowcut=2, highcut=8.5, order=4)


    return returndata,baseline

def Statedetect(data,threshold):
    """
    函数说明：
    将输入生理信号进行处理，移除大体动以及空床状态，只保留正常睡眠
    :param data:                输入信号数据
    :param threshold:           设置空床门槛
    :return:                    返还剩余的正常睡眠信号
    """
    win = windows(data, 200)
    SD = np.zeros( win.shape[0] )
    state = []

    for i in range( win.shape[0]):
        SD[i] = np.std(np.array(win[i]), ddof=1)
    Median_SD = np.median(SD)

    for i in range( len(SD) ) :

        if SD[i] > (Median_SD*2.3) :
            state.append("Movement")
        elif SD[i] < threshold :
            state.append("Movement")
        else :
            state.append("Sleep")

    return  np.array( state )

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
        if count != num * 200 :
            cutdata.append( data[count:num*200] )
            mark.append(count)
        else:
            pass
        count = (num+1)*200
    if Movestate[-1] == int(len(data)/200-1) :
        pass
    else:
        cutdata.append( data[(Movestate[-1]+1)*200: ] )
        mark.append( (Movestate[-1]+1)*200 )
    return cutdata,mark

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
            test.append(data[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))])
    meanBCG = np.zeros(ModelLength)        # ----------------------对初始预判J峰的信号段相加平均
    for num in range(len(test)):
        meanBCG += test[num]
    meanBCG = meanBCG / len(test)
    dit = np.array([])                     # ----------------------计算初始预判信号与平均信号的相似性
    for num in range(len(test)):
        dit = np.append(dit, distEuclidean(test[num], meanBCG) * 1)

    indexmin = np.array([])                # -----------------------选择与平均信号最相似的2个原始信号
    for num in range(3):
        if len(dit)>1 :
            indexmin = np.append( indexmin, np.argmin(dit) )
            dit[np.argmin(dit)] = float("inf")
        else:
            pass
    indexmin = indexmin.astype(int)
    Model = np.zeros(ModelLength)

    for num in indexmin:
        Model += test[num]
    Model = Model/3

    #--------------------------------------
    #plt.figure(figsize=(10, 6))
    #ax = plt.subplot(1, 1, 1)
    #plt.plot(np.linspace(0, len(ECG) / 1000, len(ECG)), ECG * 3.3 / 4096, 'darkgrey', label='ECG')
    #plt.plot(np.linspace(0, len(data) / 1000, len(data)), data*3.3/4096,'green',label='BCG')
    #plt.plot(np.linspace(0, len(ChooseBCG) / 1000, len(ChooseBCG)), ChooseBCG*3.3/4096,'blue',label='Selected heartbeat')
    #plt.plot(np.linspace(0, len(chooseJ) / 1000, len(chooseJ)), chooseJ*3.3/4096,'r.',markersize = 12,label='Initial estimate J-peak')
    #plt.tick_params(labelsize=18)
    #font2 = {'family': 'Times New Roman',
    #         'weight': 'normal',
    #         'size': 22,
    #         }
    #plt.xlabel('Time (s)', font2)
    #plt.ylabel('Amplitude (Volt)', font2)
    #plt.xticks([x for x in range(41)])
    #plt.legend(fontsize=12)
    # plt.show()
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
    a = []
    mini_index = np.array( a )
    # 获取峰峰值点对应的x坐标
    for i in range(1, len(data) - 2):
        mini[i] = np.where([(data[i] - data[i - 1] < 0) & (data[i] - data[i + 1] < 0)],data[i], np.nan)
        if np.isnan(mini[i]):
            continue
        mini_index = np.append(mini_index,i)
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

def BeatChoose(cor_f, cor_b, dit_f, dit_b, initInterval):
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
        else:
            beat = np.array([
                cor_f[num0],cor_b[num1],
                dit_f[num2],dit_b[num3]
            ])

            # 移除间隔小于500的点
            beat_detect = np.array([0, 1, 2, 3])
            if len(BeatPosition)>0 :
                beat_detect = np.where( (beat - BeatPosition[-1])<50 )[0]
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
            Minibeat = np.min(beat)
            beat_choose = np.where( (beat - Minibeat)<=(initInterval/2) )[0]

            case1 = False           #cor b=f
            case2 = False           #dit b=f

            if 1 and 0 in beat_choose:
                if abs(beat[0] - beat[1]) < 10 :
                    case1 = True
                else:
                    case1 = False
            else:
                case1 = False

            if 2 and 3 in beat_choose:
                if abs(beat[2] - beat[3]) < 10 :
                    case2 = True
                else:
                    case2 = False
            else:
                case2 = False

            beat = np.array(beat[beat_choose])

            if len(beat)>0 :

                if case2 and case1 :
                    pos = np.mean(beat)
                elif case1 :
                    pos = beat[0]
                elif case2 :
                    pos = beat[-1]
                else:
                    beat = beat.astype(int)
                    if len(beat)==1 :                        #长度为1时，取该点BCGCor和前3个心跳的BCGCor相比较,再和前面的RR间期相比较
                        pos = beat[0]
                    else :
                        pos = np.mean(beat)                                      #取平均作预估心跳点
                #----------判断间期是否接受
                if pos==0 :
                    pass
                elif len(BeatPosition)==1 :
                    if 50<pos-BeatPosition[-1]<150:
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                elif len(BeatPosition)>1 :
                    Interval = BeatPosition[-1] - BeatPosition[-2]
                    if 50<pos-BeatPosition[-1]<150:
                        BeatPosition = np.append(BeatPosition,pos)
                    elif (pos-BeatPosition[-1]>160):
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

#定义全局参数
Sample_org  = 100
Modellength = 70


if __name__ == '__main__':

    #打开文件
    # ECG        = np.array( openfile("./test/ECG_5min_down.txt") )
    # ECGRRI     = np.array( openfile("./in_data/10/RR.txt"))
    # orgBCG     = np.array( openfile("./test/orgData_5min_down.txt") )
    # label      = np.array( openfile("./in_data/10/label_down.txt") )*400
    # ECGRRI     = pd.read_csv("./test/ECG_RRI_down.txt",header=None).to_numpy().reshape(-1)
    #BCGRRI     = pd.read_csv("./test/BCG_RRI_down.txt",header=None).to_numpy().reshape(-1)
    test_ecg = r"D:\Studysoftware\SQI\new_data\ecg_100_0.txt"
    test_bcg = r"D:\Studysoftware\SQI\new_data\bcg_100_0.txt"

    ECG = pd.read_csv(test_ecg, delimiter='\t', header=None).to_numpy()[:, 0]
    orgBCG = pd.read_csv(test_bcg, delimiter='\t', header=None).to_numpy()[:, 0]

    ECG = ECG/20
    orgBCG = (orgBCG-1800)

    # 将信号切分为10s段，前后覆盖1s，每段12s
    orgBCG_win10s = windows_10(orgBCG, 1000)
    ECG_win10s    = windows_10(ECG,    1000)
    #ECGRRI_win10s = windows_10(ECGRRI, 1000)
    #------------------------------------------------------------------------------
    BCG  = [ [] for x in range(len(orgBCG_win10s)) ]
    Resp = [ [] for x in range(len(orgBCG_win10s)) ]

    #------------------------------------------------------------------------------
    # 需要观察的信息
    AllBCG = np.array([])
    AllResp = np.array([])
    AllBCGcor = np.array([])
    AllBCGdit = np.array([])
    AllBeat = np.array([])
    AllRRI = np.array([])
    for win in range(len(orgBCG_win10s)):
        # ------------------------------------------------------------------------------
        # -----------------------------------1.信号预处理---------------------------------
        state = Statedetect(orgBCG_win10s[win], 0.1)

        BCG[win], Resp[win] = Preprocessing2( orgBCG_win10s[win] )
        #------------------------------------2.状态检测-----------------------------------
        BCGcut, Cutmark= CutData(BCG[win], state)              #按体动分开

        #--------------------------------3.Model Formation------------------------------
        InitPeak = []
        for num in range(len(BCGcut)):
            InitPeak.extend( Cutmark[num] + Investiage.Initpeak(BCGcut[num]) )

        Model = Modeldetect(BCG[win], Modellength, InitPeak)
        #-------------------------------4.相关函数和形态距计算------------------------------
        print("cor start:" + str(datetime.datetime.now()))
        BCGcor = np.correlate(np.array(BCG[win]), np.array(Model), "same")
        print("cor end:" + str(datetime.datetime.now()))

        print("dit start:" + str(datetime.datetime.now()))
        BCGdit = []
        for j in range(len(BCG[win]) - len(Model)):
            # para = 2-ASD(xindata[win][j:j + len(ModelBCG[win])], ModelBCG[win])/SAD(xindata[win][j:j + len(ModelBCG[win])], ModelBCG[win])
            para = 1
            BCGdit.append(distEuclidean(BCG[win][j:j + len(Model)], Model) * para)
        BCGdit = np.array(BCGdit)
        BCGdit = np.insert(BCGdit, 0, np.full(int(Modellength / 2), BCGdit[0]))
        BCGdit = np.append(BCGdit, np.full(int(Modellength / 2), BCGdit[-1]))
        print("dit end:" + str(datetime.datetime.now()))

        #------------------------------------5.定位心跳-----------------------------------

        BCGcor_cut, cormark = CutData(BCGcor, state)
        BCGdit_cut, ditmark = CutData(BCGdit, state)
        #------------------------------相关
        beatcor_forward = np.array([])
        beatcor_backward = np.array([])
        for num in range(len(BCGcor_cut)):
            # 求包络线
            hx = fftpack.hilbert( BCGcor_cut[num] )
            hy = np.sqrt(BCGcor_cut[num]**2 + hx**2)
            hy = Butterworth(hy, type="lowpass", lowcut=1.5, order=4)
            # 检测位置
            cor_forward = Investiage.BeatDetection2(BCGcor_cut[num], hy, 90, up=1.6, down=0.625, style="peak")

            cor_backward = Investiage.BeatDetection2(BCGcor_cut[num][::-1], hy[::-1], 90, up=1.6, down=0.625, style="peak")
            cor_backward = len(BCGcor_cut[num]) - np.array(cor_backward)
            cor_backward = np.sort( cor_backward )
            # 组合
            beatcor_forward = np.append(beatcor_forward, Cutmark[num] + np.array(cor_forward)).astype(int)
            beatcor_backward = np.append(beatcor_backward, Cutmark[num] + cor_backward).astype(int)

            ## 删除错判峰
            #meanBCG = np.mean(np.array(BCGcor[beatcor_forward]))
            #if BCGcor[beatcor_forward[-1]] <meanBCG*0.5 :
            #    beatcor_forward = np.delete(beatcor_forward, -1)
            ## 删除错判峰
            #meanBCG = np.mean(np.array(BCGcor[beatcor_backward]))
            #if BCGcor[beatcor_backward[-1]] < meanBCG * 0.5:
            #    beatcor_backward = np.delete(beatcor_backward, -1)

        #---------------------------------形态距
        beatdit_forward = np.array([])
        beatdit_backward = np.array([])
        # 移除基线
        for num in range(len(BCGdit_cut)):
            BCGdit_cut[num] = np.max(BCGdit) - BCGdit_cut[num]
            #BCGdit_cut[num] = Preprocessing2(BCGdit_cut[num])[0]

        for num in range(len(BCGdit_cut)):
            # 求包络线
            hx = fftpack.hilbert(BCGcor_cut[num])
            hy = np.sqrt(BCGcor_cut[num] ** 2 + hx ** 2)
            hy = Butterworth(hy, type="lowpass", lowcut=1.5, order=4)
            # 检测位置
            dit_forward = Investiage.BeatDetection2(BCGdit_cut[num], hy, 90, up=1.6, down=0.625, style="peak")

            dit_backward = Investiage.BeatDetection2(BCGdit_cut[num][::-1], hy[::-1], 90, up=1.6, down=0.625, style="peak")
            dit_backward = len(BCGdit_cut[num]) - np.array(dit_backward)
            dit_backward = np.sort(dit_backward)

            # 组合
            beatdit_forward = np.append(beatdit_forward, Cutmark[num] + np.array(dit_forward)).astype(int)
            beatdit_backward = np.append(beatdit_backward, Cutmark[num] + dit_backward).astype(int)

            Corpos = np.full(len(BCG[win]), np.nan)
            for num in beatdit_backward.astype(int):
                Corpos[num] = BCGdit[num]
#
        #    ## 删除错判峰
        #    #meanBCG = np.mean(np.array(BCGdit[beatdit_forward]))
        #    #if BCGdit[beatdit_forward[-1]] < meanBCG * 0.5:
        #    #    beatdit_forward = np.delete(beatdit_forward, -1)
        #    ## 删除错判峰
        #    #meanBCG = np.mean(np.array(BCGdit[beatdit_backward]))
        #    #if BCGdit[beatdit_backward[-1]] < meanBCG * 0.5:
        #    #    beatdit_backward = np.delete(beatdit_backward, -1)
#
        ##-----------------------------------联合统一前向后向---------------------------------------
        print("OK")
        BeatPosition = BeatChoose(beatcor_forward, beatcor_backward, beatdit_forward, beatdit_backward, 90).astype(int)


        #---------------------------------------END---------------------------------------------

        # -----------------------------------------------标记展示区

        InitPeak = np.array(InitPeak).astype(int)
        Initpos = np.full(len(BCG[win]),np.nan)
        for num in InitPeak:
            Initpos[num] = BCG[win][num]

        Corpos_for = np.full(len(BCG[win]), np.nan)
        for num in beatcor_forward.astype(int):
            Corpos_for[num] = BCGcor[num]

        Corpos_back = np.full(len(BCG[win]), np.nan)
        for num in beatcor_backward.astype(int):
            Corpos_back[num] = BCGcor[num]

        #Ditpos_for = np.full(len(BCG[win]), np.nan)
        #for num in beatdit_forward.astype(int):
        #    Ditpos_for[num] = BCGdit[num]
#
        #Ditpos_back = np.full(len(BCG[win]), np.nan)
        #for num in beatdit_backward.astype(int):
        #    Ditpos_back[num] = BCGdit[num]

        DecisionBeat = np.full(len(BCG[win]), np.nan)
        for num in BeatPosition.astype(int):
            DecisionBeat[num] = BCG[win][num]*1000


        # -----------------------------------------------片段绘图区

        #bx = plt.subplot(2, 1, 1, sharex=ax)
        #bx.spines['right'].set_visible(False)
        #bx.spines['top'].set_visible(False)
        #plt.plot(np.linspace(0, len(ECG_win10s[win]) / 1000, len(ECG_win10s[win])), ECG_win10s[win], 'darkgrey',label='ECG')
        #Ditlabel = plt.plot(np.linspace(0, len(BCGdit) / 1000, len(BCGdit)), BCGdit,label='Mor.')
        #DitFlabel = plt.plot(np.linspace(0, len(Ditpos_for) / 1000, len(Ditpos_for)), Ditpos_for, '.',color='green',label='FD Mor.',markersize=12)
        #DitBlabel = plt.plot(np.linspace(0, len(Ditpos_back) / 1000, len(Ditpos_back)), Ditpos_back,'x',color='green',label='BD Mor.',markersize=15)
        #plt.tick_params(labelsize=18)
        #font2 = {'family': 'Times New Roman',
        #         'weight': 'normal',
        #         'size': 22,
        #         }
        ##plt.xlabel('Time (s)', font2)
        #plt.ylabel('Amplitude', font2)
        #plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        #lns = Corlabel+Ditlabel+ECGlabel+CorFlabel+CorBlabel+HPlabel+DitFlabel+DitBlabel
        ##lns = Corlabel + Ditlabel + HPlabel + CorFlabel  + DitFlabel
        #labs = [l.get_label() for l in lns]
        #plt.legend(lns, labs, fontsize=12,bbox_to_anchor=(1.0, 1.35),ncol=3,loc = 1)
        #plt.show()
        #if win==2900006 or win==100009 or win==200004 or win==200008 or win==200009 :
        #plt.figure()
        #plt.subplot(2, 1, 2)
        #plt.plot(np.linspace(0, len(ECG_win30s[win]) / 1000, len(ECG_win30s[win])), ECG_win30s[win], "darkgrey",label='ECG')
        #plt.plot(np.linspace(0, len(BCGcor) / 1000, len(BCGcor)), BCGcor,label='Cor.')
        #plt.plot(np.linspace(0, len(Corpos_for) / 1000, len(Corpos_for)), Corpos_for, 'r.')
        #plt.plot(np.linspace(0, len(Corpos_back) / 1000, len(Corpos_back)), Corpos_back, 'bx')
        #plt.plot(np.linspace(0, len(DecisionBeat) / 1000, len(DecisionBeat)), DecisionBeat, 'b*')
        #plt.legend()
        #plt.subplot(2, 1, 1)
        #plt.plot(np.linspace(0, len(ECG_win30s[win]) / 1000, len(ECG_win30s[win])), ECG_win30s[win], "darkgrey", label='ECG')
        #plt.plot(np.linspace(0, len(BCGdit) / 1000, len(BCGdit)), BCGdit,label='Mor.')
        #plt.plot(np.linspace(0, len(Ditpos_for) / 1000, len(Ditpos_for)), Ditpos_for, 'r.',label='forward detection')
        #plt.plot(np.linspace(0, len(Ditpos_back) / 1000, len(Ditpos_back)), Ditpos_back, 'bx',label='backward detection')
        #plt.legend()
        #plt.figure()
        #plt.subplot(1, 1, 1)
        ##plt.plot(range(len(ECG_win30s[win])), ECG_win30s[win], "darkgrey",label='ECG')
        #plt.plot(range(len(BCG[win])), BCG[win]*1000,"orange",label='BCG')
        #plt.plot(range(len(BCGcor)), BCGcor)
        #plt.plot(range(len(Corpos_for)), Corpos_for,'r.',label='Jpeak')
        #plt.plot(range(len(DecisionBeat)), DecisionBeat, 'rx', label='Jpeak')
        #plt.legend()
        ##plt.figure()
        ##plt.subplot(1, 1, 1)
        ##plt.plot(range(len(Model)), Model)
        #plt.show()
        print(win)

        # -----------------------------------------------总体合成区
        if win == 0 :
            BeatPosition = [x for x in BeatPosition if x < 1000]
        elif win==1:
            BeatPosition = [x for x in BeatPosition if 50<x<1150]
            BeatPosition = np.array(BeatPosition) + win*1000 - 100
        else:
            BeatPosition = [x for x in BeatPosition if 80<x<1130]
            BeatPosition = np.array(BeatPosition) + win*1000 - 100
        #------------
        if win == 0 :
            AllBCG = np.append(AllBCG,    BCG[win][: 1000])
            AllResp = np.append(AllResp, Resp[win][: 1000])
            AllBCGcor = np.append(AllBCGcor, BCGcor[: 1000])
            AllBCGdit = np.append(AllBCGdit, BCGdit[: 1000])
            AllBeat = np.append(AllBeat, BeatPosition)

        else:
            AllBCG = np.append(AllBCG,    BCG[win][100: 1100])
            AllResp = np.append(AllResp, Resp[win][100: 1100])
            AllBCGcor = np.append(AllBCGcor, BCGcor[100: 1100])
            AllBCGdit = np.append(AllBCGdit, BCGdit[100: 1100])
            AllBeat = np.append(AllBeat, BeatPosition)

    #--------------------------------------输出txt-------------------------------------------
    #d = writefile("./out_data/data6/BCG.txt",np.array(AllBCG))
    #d = writefile("./out_data/data6/Resp.txt", np.array(AllResp))
    #d = writefile("./out_data/data6/Beat.txt", np.array(AllBeat))

    #--------------------------------------形成RRI-------------------------------------------
    AllBeat = AllBeat.astype(int)
    AllBeatdiff = np.diff(AllBeat)
    diffindex = np.where(AllBeatdiff<50)[0]
    for i in reversed(diffindex) :
        AllBeat[i] = int( (AllBeat[i] + AllBeat[i+1])/2 )
        AllBeat = np.delete(AllBeat, i+1)

    RRI = np.full(len(AllBCG)+150,np.nan)
    for num in range(len(AllBeat) - 2):
        if 50<(AllBeat[num + 2] - AllBeat[num + 1]) < 151:
            RRI[AllBeat[num + 1]:AllBeat[num + 2]] = np.full(AllBeat[num + 2] - AllBeat[num + 1],
                                                                AllBeat[num + 1] - AllBeat[num])
        else:
            pass

    AllBeat = AllBeat.astype(int)
    PosDisplay = np.full(len(AllBCG), np.nan)
    predict = np.zeros(len(AllBCG))
    for num in AllBeat :
        PosDisplay[num] = AllBCG[num]
        predict[max(0,num-4): min(len(AllBCG),num+4)] = np.ones(min(len(AllBCG),num+4)-max(0,num-4))

    # ----------------------------------------------
    # 删除np.nan
    RRICount = np.delete(RRI, np.argwhere(np.isnan(RRI)))
    #referRRICount = np.delete(ECGRRI, np.argwhere(np.isnan(ECGRRI)))
    # 差分
    RRICount_diff = np.diff(RRICount)
    #referRRICount_diff = np.diff(referRRICount)
    # 找出不是0的下标
    RecordNo0_RRI = np.argwhere(RRICount_diff != 0)
    #RecordNo0_ECG = np.argwhere(referRRICount_diff != 0)
    #
    beatRRI = RRICount[RecordNo0_RRI]
    #beatECG = referRRICount[RecordNo0_ECG]

    beatRRI = np.append(beatRRI, RRICount[-1])
    #beatECG = np.append(beatECG, referRRICount[-1])

    #d = writefile("./out_data/new/dataallRR.txt", beatRRI)
    #b = writefile("./out_data/new/xuruibinECGRRI3.txt", beatECG)
    #pd.DataFrame(RRI).to_csv("./test/ECG_RRI_down.txt", index=False, header=False)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(range(len(ECG)), ECG*2, 'darkgrey', label='ECG')
    plt.plot(range(len(AllBCG)), AllBCG*2, label='BCG')
    plt.plot(range(len(PosDisplay)), PosDisplay*2, 'r.', label='Jpeak')
    plt.plot(range(len(RRI)), RRI, color='red',  linestyle='--', label='BCGRRI')
    #plt.plot(range(len(ECGRRI)), ECGRRI, color='green',linewidth=2,linestyle="--", label='ECG_RRI')
    #plt.plot(label,'red')
    #plt.plot(predict*400, 'grey')
    plt.legend()
    plt.show()









