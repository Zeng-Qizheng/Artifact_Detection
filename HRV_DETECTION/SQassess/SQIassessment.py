# _*_ coding: utf-8 _*_

"""
@ date:             2020-09-20
@ author:           jingxian
@ illustration:     Signal quality assessment
"""

import math

import numpy as np
import pandas as pd

from scipy import signal
from scipy.stats.stats import pearsonr

def bSQI(beat1, beat2):
    """
    函数说明：
    :param beat1:      方法1检测的心跳位置
    :param beat2:      方法2检测的心跳位置
    :return:           bSQI指标( bSQI = N_match/(N1+N2-N_match) )
    """
    if beat2.shape[0] == 0 or beat1.shape[0] == 0 : # 假如其中有一个为空，返回0
        return 0
    num_match = 0
    for b in beat1 :
        if np.min(abs(beat2 - b)) < 50 :
            num_match += 1
    return num_match/(beat1.shape[0] + beat2.shape[0] - num_match)

def tSQI(data, beat):
    """
    函数说明：计算信号段内两两心跳的相关系数来评估信号质量
    :param data:            输入信号段
    :param beat:            输入心搏位置
    :return:                tSQI ( tSQI = \sum{ c_ij } i,j 属于 [0,M) )
    """
    beat_segment = []
    for index in beat :
        if index < 350 or index > len(data)-350 :
            continue
        beat_segment.append( data[index-350: index+350])
    if len(beat_segment) <= 1 : # 若检测到的心搏数少于2个，直接返回0
        return 0
    else :
        corrcoef = np.corrcoef(beat_segment)        # 计算两两心搏的相关系数
        return np.sum(corrcoef)/(corrcoef.shape[0]**2)

def iSQI(beat):
    """
    函数说明：计算窗内升序的JJ间期分布来计算信号质量，排名第15%和排名第85%的比值
    :param beat:              算法检测的心搏位置
    :return:                  iSQI( iSQI = JJ_15/JJ_85 )
    """
    if len(beat) <= 2 :
        return 1
    beat_interval = np.diff(beat)           # 计算JJ间期
    beat_interval = np.sort(beat_interval)  # 升序排列
    index_15, index_85 = len(beat)*0.15, len(beat)*0.85
    return beat_interval[int(index_15)]/beat_interval[int(index_85)]

def aSQI(data, threshold):
    """
    函数说明： 计算信号段内，超出幅度阈值的非重叠窗口的占比量来评估信号质量
    :param data:                   输入信号段
    :param threshold:              幅度阈值
    :return:                       aSQI  ( aSQI = exp( -(Tn/5)^2 )
    """
    data_windows = data.reshape(-1, 200)    # 按论文所分窗，每个窗0.2s，不重叠
    num = 0
    for win in data_windows :
        if np.max(win) > threshold :
            num += 1
    return math.exp(-((num/5)**2))

def skSQI(data):
    """
    函数说明：统计信号的三/四阶分布(即 偏度skewness / 峰度 kurtosis)来评估信号质量
    :param data:             输入信号
    :return:                 sSQI, kSQI
    """
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    return np.abs(np.mean( ((data-mu)/sigma)**3 )), np.mean( ((data-mu)/sigma)**4 )

def SNR(data, beat):
    """
    :param data:            输入信号数据
    :param beat:            输入定位的心搏
    :return:                输入信噪比
    """
    BCG = [ [] for i in range(5) ]
    BCG_all = []
    for b in beat:
        if b < 300 or b > len(data)-300 : continue
        elif b <= 5000 :  BCG[0].append( data[b-300: b+300] )
        elif b <= 10000 : BCG[0].append( data[b-300: b+300] ); BCG[1].append( data[b-300: b+300] )
        elif b <= 15000 : BCG[1].append( data[b-300: b+300] ); BCG[2].append( data[b-300: b+300] )
        elif b <= 20000 : BCG[2].append( data[b-300: b+300] ); BCG[3].append( data[b-300: b+300] )
        elif b <= 25000 : BCG[3].append( data[b-300: b+300] ); BCG[4].append( data[b-300: b+300] )
        else: BCG[4].append( data[b-300: b+300] )
        BCG_all.append(data[b-300: b+300])
    BCG_mean = np.array([np.mean(bcg, axis=1) for bcg in BCG])
    BCG_all = np.mean(BCG_all, axis=0)
    result = 0
    for i in range(len(BCG_mean)):
        result += np.var( (BCG_all - BCG_mean)**2 )
    return result

def cor_JJ_HH(data, beats):
    """
    :param data:                输入信号数据
    :param beats:               定位的每个心搏信号
    :return:                    JJ间期和HH间期的相关系数
    """
    if len(beats) <= 1 : raise ValueError("The input beat array is %s, too small!"%( len(beats) ))
    Hpeak = []
    for beat in beats:
        Hpeak.append( np.argmax(data[beat-300:beat-100]) + beat - 300 )
    JJ_interval = np.diff( beats )
    HH_interval = np.diff( Hpeak )
    try :
        result = pearsonr(JJ_interval, HH_interval)
    except Exception as e :
        print(e)
        return None
    else:
        return result



