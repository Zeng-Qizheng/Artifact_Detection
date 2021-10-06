# _*_ coding: utf-8 _*_


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import math
import copy
import datetime

from scipy import signal
from scipy import fftpack
from scipy import interpolate


Sample_org = 1000



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
        Maxrange = min(len(data), peak + 200)
        chooseV = [int(x) for x in BCG_Vindex if Minrange < x < Maxrange]
        if len(chooseV) == 0:
            continue
        chooseP = np.full(len(chooseV), data[peak])
        MaxAm = np.max(chooseP - data[chooseV])
        Amp_JK[int(peak)] = MaxAm
    index = np.where(Amp_JK != 0)[0]
    return np.array(Amp_JK[index]),index #返回片段内每个峰的JK的幅值差和对应的峰值索引

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
                while (dit > 1100) or (dit < 650) and len(peak)>1 :
                    if dit >1100 :
                        EnvelopPeak = findpeak(envelop[index[-1]:beat]) + index[-1]
                        EnvelopPeak = np.array( [x for x in EnvelopPeak if x > win_min+100] )
                        if len( EnvelopPeak ) == 0 :
                            peak = np.delete(peak, peakmax)
                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            if data[Senbeat] > data[beat] * 0.75:
                                beat = Senbeat
                            else:
                                break
                            dit = beat - index[-1]
                        else:
                            peak = np.delete(peak, peakmax)
                            peak = np.array([x for x in peak if x < (EnvelopPeak[-1] + 350)])
                            peak = peak.astype(int)
                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            if data[Senbeat] > data[beat] * 0.75:
                                beat = Senbeat
                            else:
                                break
                            dit = beat - index[-1]
                    elif dit < 650 :
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])
                        Senbeat = peak[peakmax]
                        if data[Senbeat] >data[beat]*0.75 :
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

def BeatDetection2(data,envelop,MeanRR,up=1.86,down=0.53,style="peak"):
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
            peak = findpeak(data[win_min:win_max]) + win_min    #寻找1.5s片段内信号峰值，返回对应的坐标索引
            peak = peak.astype(int)
            if len(peak) == 0:
                break
            peakmax = np.argmax(data[peak])   #根据1.5s片段的索引，找到原信号峰值索引
            beat = peak[peakmax]
            if len(index)==0 :               #首个检测
                if (beat>MeanRR*1.2) and len(peak)>1 :          #如果第一个峰值的位置索引大于MeanRR*1.2，查询包络在范围[win_min:win_min + 1000]内是否有峰值
                    EnvelopPeak = findpeak(envelop[win_min:win_min + 1000]) + win_min
                    if len( EnvelopPeak ) == 0 :
                        index.append(beat)  #如果包络没有峰值，则直接保存该峰值的索引
                    else:  #如果包络检测到峰值，则在原信号范围[win_min:win_min + 1000]内重新找峰值
                        peak = findpeak(data[win_min:win_min + 1000]) + win_min
                        peak = peak.astype(int)
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        index.append(beat) #保存峰值的位置索引
                else:
                    index.append(beat)  #如果第一个峰值的位置索引小于MeanRR*1.2，则直接保存
            else: #检测剩余点
                dit = beat - index[-1]  #当前峰值索引与上一峰值索引的距离dit
                if len(index)==1 :
                    std = MeanRR    #如果当前检测第二个点，则平均间期为MeanRR
                else:  #如果当前检测点>=3 ，则平均间期前面检测点间期的均值
                    Interval = np.diff(index)
                    Interval = np.append(Interval,MeanRR)
                    std = np.mean(Interval)
                    # std = np.mean( np.append(np.diff(index),MeanRR) )
                    # std = MeanRR
                while ((dit > (std*1.4)) or (dit < (std/1.4))) and (len(peak)>1) :  #如果当前间期dit大于std*1.4或小于std/1.4，则需要重新检测
                    if (dit > (std*1.4)) : #如果当前间期dit大于std*1.4，则在范围[win_min:win_min+500]重新找点
                        EnvelopPeak = findpeak(envelop[win_min:win_min + 500]) + win_min
                        if len( EnvelopPeak ) == 0 :
                            peak = np.delete(peak, peakmax)
                            peakmax = np.argmax(data[peak])#如果包络重检测不到点，则删除该峰值点的索引，寻找次峰值点的索引
                            Senbeat = peak[peakmax]
                            if data[Senbeat] > data[beat] * 0.9: #如果次峰值的大小大于0.9倍的最大峰值，则保存次峰值点的索引
                                beat = Senbeat
                            else:
                                break
                            dit = beat - index[-1]
                            data[beat] = np.nan
                        else:
                            peak = findpeak(data[win_min:win_min + 500]) + win_min #如果包络重检测到新的峰值，则保存新检测到的峰值
                            peak = peak.astype(int)
                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            beat = Senbeat
                            dit = beat - index[-1]
                            data[beat] = np.nan

                    elif dit < (std/1.4) :  #如果当前间期dit太小
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])  #则删除当前峰值，寻找次峰值
                        Senbeat = peak[peakmax]
                        if data[Senbeat] >data[beat]*0.9 :   #如果次峰值的大小大于0.9倍的最大峰值，则保存次峰值点的索引
                            beat = Senbeat
                        else:
                            break
                        dit = beat - index[-1]
                        data[beat] = np.nan
                    elif dit > up*std or dit < down*std :    #如果当前间期dit超过上限或下限
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])    #则直接保存次峰值点的索引
                        beat = peak[peakmax]
                        dit = beat - index[-1]
                        data[beat] = np.nan
                    else:
                        break

                index.append(beat)
            win_min = max(0, index[-1] + 500)   #添加不应期500ms
            win_max = index[-1] + 1500
            if (win_max > length) :
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

def BeatDetection3(data,envelop,MeanRR,up=1.86,down=0.53,style="peak"):
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
            copy_data = np.copy(data) #备份data，因为后面会对data值修改
            Amp_JK,peak = Amp_JKDetection(data[win_min:win_max])     #寻找1.5s片段内信号峰值，返回对应的坐标索引
            peak = peak + win_min
            peak = peak.astype(int)
            if len(peak) == 0:
                 break
            peakmax = np.argmax(Amp_JK)   #根据1.5s片段的索引，找到原信号峰值索引
            beat = peak[peakmax]
            if len(Amp_JK) and len(peak) >1 :#看范围内是否有JK值接近的第二个峰
                copy_Amp_JK = np.copy(Amp_JK) #保存最开始找到的峰值索引
                copy_peak = np.copy(peak)
                copy_Amp_JK= np.delete(copy_Amp_JK,peakmax)
                copy_peak = np.delete(copy_peak, peakmax)
                indics_peakmax = np.argmax(copy_Amp_JK)
                indics_beat =  copy_peak[indics_peakmax]     #JK值接近的候选峰
                Indics = 0  # 是否有JK值接近的候选峰
                if  abs(Amp_JK[peakmax] - Amp_JK[indics_peakmax]) < 1000:
                    Indics = 1 #Indics表示有峰值接近的候选峰
            else:
                Indics = 0
                indics_beat = beat

            if len(index)==0 :               #首个检测
                if (beat>MeanRR*1.2) and len(peak)>1 :          #如果第一个峰值的位置索引大于MeanRR*1.2，查询包络在范围[win_min:win_min + 1000]内是否有峰值
                    Amp_JK,peak = Amp_JKDetection(envelop[win_min:win_min + 1000])
                    if len( Amp_JK ) == 0 :
                        index.append(beat)  #如果包络没有峰值，则直接保存该峰值的索引
                    else:  #如果包络检测到峰值，则在原信号范围[win_min:win_min + 1000]内重新找峰值
                        Amp_JK,peak = Amp_JKDetection(data[win_min:win_min + 1000])
                        peak = peak + win_min
                        peak = peak.astype(int)
                        peakmax = np.argmax(Amp_JK)
                        beat = peak[peakmax]
                        index.append(beat) #保存峰值的位置索引
                else:
                    index.append(beat)  #如果第一个峰值的位置索引小于MeanRR*1.2，则直接保存
            else: #检测剩余点
                dit = beat - index[-1]  #当前峰值索引与上一峰值索引的距离dit
                if len(index)==1 :
                    std = MeanRR    #如果当前检测第二个点，则平均间期为MeanRR
                else:  #如果当前检测点>=3 ，则平均间期前面检测点间期的均值
                    Interval = np.diff(index)
                    Interval = np.append(Interval,MeanRR)
                    std = np.mean(Interval)
                    # std = np.mean( np.append(np.diff(index),MeanRR) )
                    # std = MeanRR
                while ((dit > (std*1.4)) or (dit < (std/1.4))) and (len(peak)>1) :  #如果当前间期dit大于std*1.4或小于std/1.4，则需要重新检测
                    if (dit > (std*1.4)) : #如果当前间期dit大于std*1.4，则在范围[win_min:win_min+500]重新找点
                        Amp_JK,peak = Amp_JKDetection(envelop[win_min:win_min + 500])
                        if len( Amp_JK ) == 0 :
                            # del_Amp_JK= np.copy(Amp_JK[peakmax])#保存当前最大值索引后，删除重新找
                            # Amp_JK = np.delete(Amp_JK,peakmax)
                            # peak = np.delete(peak, peakmax)
                            # peakmax = np.argmax(Amp_JK)#如果包络重检测不到点，则删除该峰值点的索引，寻找次峰值点的索引
                            # Senbeat = peak[peakmax]
                            # if data[Senbeat] > data[beat] * 0.9 or Amp_JK[peakmax] > del_Amp_JK: #如果次峰值的大小大于0.9倍的最大峰值，则保存次峰值点的索引
                            #     beat = Senbeat
                            # else:
                            #     break
                            # dit = beat - index[-1]
                            # data[beat] = np.nan
                         #如果包络找不到峰值，则直接添加该点
                            break
                        else:
                            Amp_JK,peak = Amp_JKDetection(data[win_min:win_min + 500]) #如果包络重检测到新的峰值，则保存新检测到的峰值
                            peak = peak  + win_min
                            peak = peak.astype(int)
                            peakmax = np.argmax(Amp_JK)
                            Senbeat = peak[peakmax]
                            beat = Senbeat
                            dit = beat - index[-1]
                            data[beat] = np.nan

                    elif dit < (std/1.4) :  #如果当前间期dit太小
                        del_Amp_JK= np.copy(Amp_JK[peakmax])#保存当前最大值索引后，删除重新找
                        Amp_JK = np.delete(Amp_JK, peakmax)
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(Amp_JK)  #则删除当前峰值，寻找次峰值
                        Senbeat = peak[peakmax]
                        if data[Senbeat] >data[beat]*0.9  or Amp_JK[peakmax] > del_Amp_JK:   #如果次峰值的大小大于0.9倍的最大峰值，则保存次峰值点的索引
                            beat = Senbeat
                        else:
                            beat = beat
                        dit = beat - index[-1]
                        data[beat] = np.nan
                    elif dit > up*std or dit < down*std :    #如果当前间期dit超过上限或下限
                        del_Amp_JK= np.copy(Amp_JK[peakmax])#保存当前最大值索引后，删除重新找
                        Amp_JK = np.delete(Amp_JK, peakmax)
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(Amp_JK)    #则直接保存次峰值点的索引
                        beat = peak[peakmax]
                        dit = beat - index[-1]
                        data[beat] = np.nan
                    else:
                        break
                #判断峰值和候选峰值
                if Indics ==1 and copy_data[beat] <copy_data[indics_beat] and   std/1.4 < indics_beat - index[-1] < std*1.4: #如果有候选峰值，且候选峰值的值大于原峰值，且符合间期条件，则添加候选峰值
                    index.append(indics_beat)
                else: #其他条件则直接添加原峰值
                    index.append(beat)
            win_min = max(0, index[-1] + 500)   #添加不应期500ms
            win_max = index[-1] + 1500
            if (win_max > length) :
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
                beat_detect = np.where( (beat - BeatPosition[-1])<500 )[0]
                if 0 in beat_detect:
                    num0 = num0 + 1
                if 1 in beat_detect:
                    num1 = num1 + 1
                if 2 in beat_detect:
                    num2 = num2 + 1
                if 3 in beat_detect:
                    num3 = num3 + 1
                if len(beat_detect)!=0 :
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
                    if 500<pos-BeatPosition[-1]<1500:
                        BeatPosition = np.append(BeatPosition, pos)
                    else:
                        pass
                elif len(BeatPosition)>1 :
                    Interval = BeatPosition[-1] - BeatPosition[-2]
                    if (pos-BeatPosition[-1]>Interval*0.33) and (pos-BeatPosition[-1]<Interval*3) and 500<pos-BeatPosition[-1]<1500:
                        BeatPosition = np.append(BeatPosition,pos)
                    elif (pos-BeatPosition[-1]>1600):
                        BeatPosition = np.append(BeatPosition,(pos-BeatPosition[-1])*0.5+BeatPosition[-1])
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
    data = np.insert(data, 0, np.zeros(200))
    data = np.insert(data, -1, np.zeros(200))
    #开运算:腐蚀+膨胀
    out1 = Eorde(data, len(data), g, M)     #腐蚀
    out2 = Dilate(out1, len(out1), g, M)    #膨胀
    out2 = np.insert(out2, 0, np.zeros(198))
    #闭运算:膨胀+腐蚀+腐蚀+膨胀
    out5 = Dilate(data, len(data), g, M)    #膨胀
    out6 = Eorde(out5, len(out5), g, M)     #腐蚀
    out6 = np.insert(out6, 0, np.zeros(198))

    baseline = (out2 + out6) / 2
    #-------------------------保留剩余价值------------------------
    returndata = data[:len(baseline)] - baseline
    returndata = np.delete(returndata, range(0,200), axis=0)
    returndata = returndata[:120000]
    returndata[-1] = returndata[-2] = returndata[-3]
    #-----------------------------------------------------------


    baseline = baseline[200:12200]
    baseline = Butterworth(baseline, type="bandpass", lowcut=0.01, highcut=0.7, order=2)

    return returndata,baseline

def AutoCor(BCGcor, BCGdit,state,ECG):
    """
    自相关检测峰谷值定位心跳间隔
    :param BCGcor:                  相关信号
    :param BCGdit:                  形态距信号
    :return:                        心跳位置
    """
    BCGcor_cut, cormark = CutData(BCGcor, state)
    BCGdit_cut, ditmark = CutData(BCGdit, state)

    if len(BCGcor_cut)==0 :
        return np.array([])

    # 对于dit移除基线和反转
    for num in range(len(BCGdit_cut)):
        BCGdit_cut[num] = Preprocessing2(BCGdit_cut[num])[0]


    beatcor_forward = np.array([])
    beatcor_backward = np.array([])
    # cor自相关
    for num in range(len(BCGcor_cut)):
        BCGcorMove = copy.deepcopy(BCGcor_cut[num])
        corcor = np.array([])
        for x in range(len(BCGcorMove)):
            df = pd.DataFrame({'a': BCGcorMove, 'b': BCGcor_cut[num]})
            corcor = np.append(corcor, df.corr()['a']['b'])
            BCGcorMove = np.insert(BCGcorMove, 0, BCGcorMove[-1])
            BCGcorMove = np.delete(BCGcorMove, -1)
    # 自相关后峰谷值
        Amp_cor = np.zeros(len(corcor))
        corcor_Pindex = findpeak(corcor)
        corcor_Vindex = findtrough(corcor)

        for peak in corcor_Pindex:
            Minrange = max(0, peak - 250)
            Maxrange = min(len(corcor), peak + 250)
            chooseV = [int(x) for x in corcor_Vindex if Minrange < x < Maxrange]
            if len(chooseV) == 0:
                continue
            chooseP = np.full(len(chooseV), corcor[peak])
            MaxAm = np.max(chooseP - corcor[chooseV])
            Amp_cor[int(peak)] = MaxAm

        # 根据峰谷值得到心跳位置
        cor_forward = BeatDetection(Amp_cor, np.full(len(Amp_cor),0), 900, up=1.6, down=0.625, style="peak")
        cor_backward = BeatDetection(Amp_cor[::-1], np.full(len(Amp_cor),0), 900, up=1.6, down=0.625, style="peak")
        cor_backward = len(Amp_cor) - np.array(cor_backward)
        cor_backward = np.sort(cor_backward)

        # 组合
        beatcor_forward = np.append(beatcor_forward, cormark[num] + np.array(cor_forward)).astype(int)
        beatcor_backward = np.append(beatcor_backward, cormark[num] + cor_backward).astype(int)

    # dit自相关
    beatdit_forward = np.array([])
    beatdit_backward = np.array([])
    for num in range(len(BCGdit_cut)):
        BCGditMove = copy.deepcopy(BCGdit_cut[num])
        ditcor = np.array([])
        for x in range(len(BCGditMove)):
            df = pd.DataFrame({'a': BCGditMove, 'b': BCGdit_cut[num]})
            ditcor = np.append(ditcor, df.corr()['a']['b'])
            BCGditMove = np.insert(BCGditMove, 0, BCGditMove[-1])
            BCGditMove = np.delete(BCGditMove, -1)
        # 自相关后峰谷值
        Amp_dit = np.zeros(len(ditcor))
        corcor_Pindex = findpeak(ditcor)
        corcor_Vindex = findtrough(ditcor)

        for peak in corcor_Pindex:
            Minrange = max(0, peak - 250)
            Maxrange = min(len(ditcor), peak + 250)
            chooseV = [int(x) for x in corcor_Vindex if Minrange < x < Maxrange]
            if len(chooseV) == 0:
                continue
            chooseP = np.full(len(chooseV), ditcor[peak])
            MaxAm = np.max(chooseP - ditcor[chooseV])
            Amp_dit[int(peak)] = MaxAm

            # 根据峰谷值得到心跳位置
        dit_forward = BeatDetection(Amp_dit, np.full(len(Amp_dit),0), 900, up=1.6, down=0.625, style="peak")
        dit_backward = BeatDetection(Amp_dit[::-1], np.full(len(Amp_dit),0), 900, up=1.6, down=0.625, style="peak")
        dit_backward = len(Amp_dit) - np.array(dit_backward)
        dit_backward = np.sort(dit_backward)

        # 组合
        beatdit_forward = np.append(beatdit_forward, cormark[num] + np.array(dit_forward)).astype(int)
        beatdit_backward = np.append(beatdit_backward, cormark[num] + dit_backward).astype(int)

    #----------------------------------------------联合------------------------------------------------
    BeatPosition = BeatChoose(beatcor_forward, beatcor_backward, beatdit_forward, beatdit_backward, 900).astype(int)

    #----------------------------------------------绘图------------------------------------------------

    return BeatPosition

def AmpDetect(BCGcor, BCGdit, state, ECG):
    """
    :param BCGcor:                 相关信号
    :param BCGdit:                 形态距信号
    :param state:                  检测状态
    :param ECG:                    参考ECG
    :return:                       心跳位置
    """
    BCGcor_cut, cormark = CutData(BCGcor, state)
    BCGdit_cut, ditmark = CutData(BCGdit, state)

    if len(BCGcor_cut) == 0:
        return np.array([])

    beatcor_forward = np.array([])
    beatcor_backward = np.array([])
    # 相关
    for num in range(len(BCGcor_cut)):
        # 相关峰谷值
        Amp_cor = np.zeros(len(BCGcor_cut[num]))
        Amp_corfang = np.zeros(len(BCGcor_cut[num]))
        corcor_Pindex = findpeak(BCGcor_cut[num])
        corcor_Vindex = findtrough(BCGcor_cut[num])

        for peak in corcor_Pindex:
            Minrange = max(0, peak )
            Maxrange = min(len(BCGcor_cut[num]), peak + 250)
            chooseV = [int(x) for x in corcor_Vindex if Minrange < x < Maxrange]
            if len(chooseV) == 0:
                continue
            chooseP = np.full(len(chooseV), BCGcor_cut[num][peak])
            MaxAm = np.max(chooseP - BCGcor_cut[num][chooseV])
            Amp_cor[int(peak)] = MaxAm/2000
            Amp_corfang[int(peak):] = MaxAm/1000

        # 包络
        hx = fftpack.hilbert(Amp_corfang)
        hy = np.sqrt(Amp_corfang ** 2 + hx ** 2)
        hy = Butterworth(hy, type="lowpass", lowcut=1.5, order=4)
        # 根据峰谷值得到心跳位置
        cor_forward = BeatDetection(Amp_cor, hy, 900, up=1.6, down=0.625, style="peak")
        cor_backward = BeatDetection(Amp_cor[::-1], hy[::-1], 900, up=1.6, down=0.625, style="peak")
        cor_backward = len(Amp_cor) - np.array(cor_backward) - 1
        cor_backward = np.sort(cor_backward)

        # 组合
        beatcor_forward = np.append(beatcor_forward, cormark[num] + np.array(cor_forward)).astype(int)
        beatcor_backward = np.append(beatcor_backward, cormark[num] + cor_backward).astype(int)

    # dit自相关
    beatdit_forward = np.array([])
    beatdit_backward = np.array([])
    # 移除基线
    for num in range(len(BCGdit_cut)):
        BCGdit_cut[num] = np.max(BCGdit) - BCGdit_cut[num]
        BCGdit_cut[num] = Preprocessing2(BCGdit_cut[num])[0]

    for num in range(len(BCGdit_cut)):
        # 形态距后峰谷值
        Amp_dit = np.zeros(len(BCGdit_cut[num]))
        Amp_ditfang = np.zeros(len(BCGcor_cut[num]))
        corcor_Pindex = findpeak(BCGdit_cut[num])
        corcor_Vindex = findtrough(BCGdit_cut[num])

        for peak in corcor_Pindex:
            Minrange = max(0, peak )
            Maxrange = min(len(BCGdit_cut[num]), peak + 250)
            chooseV = [int(x) for x in corcor_Vindex if Minrange < x < Maxrange]
            if len(chooseV) == 0:
                continue
            chooseP = np.full(len(chooseV), BCGdit_cut[num][peak])
            MaxAm = np.max(chooseP - BCGdit_cut[num][chooseV])
            Amp_dit[int(peak)] = MaxAm
            Amp_ditfang[int(peak):] = MaxAm

        # 包络
        hx = fftpack.hilbert(Amp_ditfang)
        hy = np.sqrt(Amp_ditfang ** 2 + hx ** 2)
        hy = Butterworth(hy, type="lowpass", lowcut=1.5, order=4)
        # 根据峰谷值得到心跳位置
        dit_forward = BeatDetection(Amp_dit, hy, 900, up=1.6, down=0.625, style="peak")
        dit_backward = BeatDetection(Amp_dit[::-1], hy[::-1], 900, up=1.6, down=0.625, style="peak")
        dit_backward = len(Amp_dit) - np.array(dit_backward)- 1
        dit_backward = np.sort(dit_backward)

        # 组合
        beatdit_forward = np.append(beatdit_forward, cormark[num] + np.array(dit_forward)).astype(int)
        beatdit_backward = np.append(beatdit_backward, cormark[num] + dit_backward).astype(int)

    # ----------------------------------------------联合------------------------------------------------
    BeatPosition = BeatChoose(beatcor_forward, beatcor_backward, beatdit_forward, beatdit_backward, 1000).astype(int)

    BCGcor = BCGcor/1000
    BCGdit = BCGdit/5
    # ----------------------------------------------绘图------------------------------------------------
    Corpos_for = np.full(len(BCGcor), np.nan)
    for num in beatcor_forward.astype(int):
        Corpos_for[num] = Amp_cor[num]

    Corpos_back = np.full(len(BCGcor), np.nan)
    for num in beatcor_backward.astype(int):
        Corpos_back[num] = Amp_cor[num]

    Ditpos_for = np.full(len(BCGcor), np.nan)
    for num in beatdit_forward.astype(int):
        Ditpos_for[num] = Amp_dit[num]

    Ditpos_back = np.full(len(BCGcor), np.nan)
    for num in beatdit_backward.astype(int):
        Ditpos_back[num] = Amp_dit[num]

    DecisionBeat = np.full(len(BCGcor), np.nan)
    for num in BeatPosition.astype(int):
        DecisionBeat[num] = 0

    #plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.plot(range(len(ECG)), ECG, "darkgrey")
    #plt.plot(range(len(BCGcor)), BCGcor)
    #plt.plot(range(len(Amp_cor)), Amp_cor)
    #plt.plot(range(len(Corpos_for)), Corpos_for, 'r.')
    #plt.plot(range(len(Corpos_back)), Corpos_back, 'bx')
    #plt.plot(range(len(DecisionBeat)), DecisionBeat, 'b*')
    #plt.subplot(2, 1, 2)
    #plt.plot(range(len(ECG)), ECG, "darkgrey")
    #plt.plot(range(len(BCGdit)), BCGdit)
    #plt.plot(range(len(Amp_dit)), Amp_dit)
    #plt.plot(range(len(Ditpos_for)), Ditpos_for, 'r.')
    #plt.plot(range(len(Ditpos_back)), Ditpos_back, 'bx')
    #plt.show()

    return BeatPosition

def Initpeak(data,style="peak"):
    """
    函数说明：
    查找合理心跳点
    :param data:                  输入数据信号
    :param maxi_index:            输入数据峰值坐标
    :return:                      处理后的峰值坐标
    """
    Amp_JK = np.zeros(len(data))
    Amp_JKfang = np.zeros(len(data))
    BCG_Pindex = findpeak(data)
    BCG_Vindex = findtrough(data)

    for peak in BCG_Pindex:
        Minrange = max(0, peak)
        Maxrange = min(len(data), peak + 300)
        chooseV = [int(x) for x in BCG_Vindex if Minrange < x < Maxrange]
        if len(chooseV) == 0:
            continue
        chooseP = np.full(len(chooseV), data[peak])
        MaxAm = np.max(chooseP - data[chooseV])
        Amp_JK[int(peak)] = MaxAm
        # Amp_JKfang[int(peak):] = MaxAm
    length = len(data)
    index = []
    # 创建包络
    hx = fftpack.hilbert(data)
    hy = np.sqrt(data ** 2 + hx ** 2)
    hy = Butterworth(hy, type="lowpass", lowcut=1, order=4)

    # 创建当前峰和识别窗
    win_min = 0
    win_max = 1500
    while (True):
        if style=="peak":
            beat = int(np.argmax( Amp_JK[win_min:win_max] ) + win_min)
        else:
            beat = int(np.argmin( Amp_JK[win_min:win_max] ) + win_min)

        if len(index)==0 : # 当首个检测时
            index.append(beat)
        elif (beat-index[-1])>1200 :
            peak = findpeak(hy[win_min:win_min + 500]) + win_min
            if  len(peak)==0 :
                index.append(beat)
            else :
                beat = int(np.argmax(Amp_JK[win_min:win_min+500]) + win_min)
                index.append(beat)
        else:
            index.append(beat)
        win_min = max(0, beat + 500)
        win_max = min(length, beat + 1500)
        if (win_min >= length-3):
            break
    return np.array(index)

def InitBeatDetect(BCG, style="peak"):
    #BCG = Butterworth(BCG, type='bandpass', lowcut=1, highcut=20, order=2)
    length = len(BCG)
    index = []
    win_min = 0
    win_max = 1000
    while(True):
        if style=="peak" :
            Peaklist = findpeak(BCG[win_min:win_max]) + win_min
            Peaklist = Peaklist.astype(int)
            if len(Peaklist)==0 :
                break
            peakmax = np.argmax(BCG[Peaklist])

            beat = np.array([])
            leftbeat = peakmax - 1
            rightbeat = peakmax + 1

            beat = np.append(beat, Peaklist[peakmax])
            temp = 0
            while(len(beat)<3):
                if leftbeat>=0 and rightbeat<len(Peaklist) :
                    if BCG[Peaklist[leftbeat]] >= BCG[Peaklist[rightbeat]] :
                        temp = Peaklist[leftbeat]
                        leftbeat = leftbeat - 1
                    else:
                        temp = Peaklist[rightbeat]
                        rightbeat = rightbeat + 1
                elif leftbeat<0 and rightbeat<len(Peaklist) :
                    temp = Peaklist[rightbeat]
                    rightbeat = rightbeat + 1
                elif leftbeat>=0 and rightbeat >= len(Peaklist) :
                    temp = Peaklist[leftbeat]
                    leftbeat = leftbeat - 1
                else:
                    break
                beat = np.append(beat, temp)
        index.append(np.median(beat))
        win_min = int(max(0,  index[-1] + 500))
        win_max = int(min(length,  index[-1] + 1500))
        if (win_min >= length-3):
            break
    return np.array(index)

def InitBeatDetect2(BCG, ECG, style="peak"):
    Amp_JK = np.zeros(len(BCG))
    Amp_JKfang = np.zeros(len(BCG))
    BCG_Pindex = findpeak(BCG)
    BCG_Vindex = findtrough(BCG)

    for peak in BCG_Pindex:
        Minrange = max(0, peak)
        Maxrange = min(len(BCG), peak + 300)
        chooseV = [int(x) for x in BCG_Vindex if Minrange < x < Maxrange]
        if len(chooseV) == 0:
            continue
        chooseP = np.full(len(chooseV), BCG[peak])
        MaxAm = np.max(chooseP - BCG[chooseV])
        Amp_JK[int(peak)] = MaxAm
        Amp_JKfang[int(peak):] = MaxAm

    hx = fftpack.hilbert(Amp_JKfang)
    hy = np.sqrt(Amp_JKfang ** 2 + hx ** 2)
    hy = Butterworth(hy, type="lowpass", lowcut=1.5, order=4)

    index = np.array(BeatDetection(Amp_JK, hy, 900 )).astype(int)
    #index = np.array(Initpeak(Amp_JK, "peak")).astype(int)
    pos = np.full(len(BCG), np.nan)
    for num in index :
        pos[num] = BCG[num]


    # ----------------------------------------
    #plt.figure()
    #plt.subplot(1,1,1)
    #plt.plot(range(len(ECG)), ECG, 'darkgrey',label='ECG')
    #plt.plot(range(len(Amp_JK)), Amp_JK,label='Amp')
    #plt.plot(range(len(hy)), hy, )
    #plt.plot(range(len(ECG)), ECG, 'darkgrey', label='ECG')
    #plt.plot(range(len(BCG)), BCG, 'orange',label='BCG')
    #plt.plot(range(len(pos)), pos, 'r.',label='Jpeak')
    #plt.legend()
    #plt.show()

    return index

