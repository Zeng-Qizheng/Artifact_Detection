# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/9
@Auth ： 曾启正 Keyon Tsang
@File ： Artifact_Detection.py
@IDE  ： PyCharm
@Motto： ABC(Always Be Coding)
@Func ： undetermined
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import os

# @vectorize(['complex64(complex64, complex64)'], target='gpu')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，否则图内标注无法显示中文
plt.rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU加速


# with open(labelPath, encoding='utf-8-sig') as file:
#     orgLabel  = csv.reader(file)
#     Label = []
#     for row in orgLabel:
#         temLabel = int(row[0])
#         Label.append(temLabel)
# Label = np.array(Label)
# print(Label.reshape(-1, 4))

# @vectorize(nopython=True, parallel = True)    #numba Python加速的一种装饰器，尚未调试成功
def dataSubsampled(sampleNum=1):
    """
    Author:Qz
    函数说明:对原时间序列进行降采样
    :param sampleNum:             输入降采样倍数
    :return:                      无
    """
    for i in range(len(orgBCG) // sampleNum):
        newBCG[i] = orgBCG[i * sampleNum]
    for i in range(orgLabel.shape[0]):
        orgLabel[i][2] //= sampleNum
        orgLabel[i][3] //= sampleNum

    print('降采样倍数：%d' % sampleNum)
    print('原始数据长度：%d' % len(orgBCG))
    print('降采样后长度：%d' % len(newBCG))
    print('体动个数：%d' % orgLabel.shape[0])


def labelShow():
    """
    Author:Qz
    函数说明:显示每个体动片段的序号和起始位置（单位：s）
    :param :              无
    :return:              无
    """
    for i in range(len(orgLabel)):
        if len(newBCG[orgLabel[i][2]:orgLabel[i][3]]) != 0:  # 有时候会出现空序列，意思是某个体动范围内有空值？无解
            plt.text((orgLabel[i][2] + orgLabel[i][3]) // 2, np.max(newBCG[orgLabel[i][2]:orgLabel[i][3]]) + 70,
                     # 用numpy的max比普通max快很多，表现在拖动图片时延迟低很多
                     orgLabel[i][0], fontsize=8, color="SteelBlue", style="italic", weight="light",
                     verticalalignment='center', horizontalalignment='center', rotation=0)
            plt.text((orgLabel[i][2] + orgLabel[i][3]) // 2, np.max(newBCG[orgLabel[i][2]:orgLabel[i][3]]) + 40,
                     (sampleNum * orgLabel[i][2] // 1000, sampleNum * orgLabel[i][3] // 1000),
                     fontsize=6, color="DarkSlateGray", style="italic", weight="light", verticalalignment='center',
                     horizontalalignment='center', rotation=0)


def dataProcessing():
    """
    Author:Qz
    函数说明:将不同类型的体动分别赋值给多个一维List
    :param :              无
    :return:              无
    """
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
        # for t in range(orgLabel[i][2], orgLabel[i][3]):
        #     orgBCG[t] = np.nan    #这样赋值会报错
        # 赋空值给正常数据List的体动部分，相当于扣掉，方便画图显示
        newBCG[orgLabel[i][2]:orgLabel[i][3]] = ArtifactData_tpye0[orgLabel[i][2]:orgLabel[i][3]]


# @jit(nopython=True) #numba Python加速的一种装饰器，尚未调试成功
def muti_dataShow(showTimes=1):
    """
    Author:Qz
    函数说明:将数据分多附图显示，但目前效果不尽人意，依旧拖动卡顿严重
    :param showTimes:     分图个数
    :return:              无
    """
    for i in range(showTimes):
        plt.plot(ArtifactData_tpye1, color='red', label="大体动")
        plt.plot(ArtifactData_tpye2, color='blue', label="小体动")
        plt.plot(ArtifactData_tpye3, color='brown', label="深呼吸")
        plt.plot(ArtifactData_tpye4, color='purple', label="脉冲体动")
        plt.plot(ArtifactData_tpye5, color='orange', label="无效片段")
        plt.plot(newBCG, color='green', label="正常数据")
        plt.legend(ncol=2)

        dataLen = len(newBCG)
        plt.xlim(dataLen * i // showTimes, dataLen * (i + 1) // showTimes)
        # plt.ylim(1550, 2250)
        # plt.savefig(filePath + "/raw_org_label.jpg", dpi=300)
        plt.show()


def dataShow(showTimes=1):
    """
    Author:Qz
    函数说明:分别画图
    :param :              无
    :return:              无
    """
    plt.plot(ArtifactData_tpye1, color='red', label="大体动")
    plt.plot(ArtifactData_tpye2, color='blue', label="小体动")
    plt.plot(ArtifactData_tpye3, color='Gold', label="深呼吸")
    plt.plot(ArtifactData_tpye4, color='purple', label="脉冲体动")
    plt.plot(ArtifactData_tpye5, color='orange', label="无效片段")
    plt.plot(newBCG, color='green', label="正常数据")
    plt.legend(ncol=2)
    # plt.ylim(1550, 2250)  #设置Y轴显示范围
    # plt.savefig(filePath + "/raw_org_label.jpg", dpi=300) #保存图片
    plt.show()


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


if __name__ == "__main__":
    filePath = "/home/qz/文档/Qz/WorkSpace/ArtifactDataset/岑玉娴2018060622-16-25"  # 文件夹的绝对路劲
    bcgPath = filePath + "/raw_org.txt"  # 原始数据路径
    labelPath = filePath + "/Artifact_a.txt"  # 体动数据路径
    orgBCG = pd.read_csv(bcgPath, header=None).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
    orgLabel = pd.read_csv(labelPath, header=None).to_numpy().reshape(-1, 4)  # 标签数据读取为numpy形式，并reshape为n行4列的数组

    orgBCG = Butterworth(orgBCG, type='bandpass', lowcut=2, highcut=5, order=2, Sample_org=1000)

    sampleNum = 20  # 降采样倍数
    newBCG = np.full(len(orgBCG) // sampleNum, np.nan)  # 创建与orgBCG降采样后一样长度的空数组
    ArtifactData_tpye1 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye2 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye3 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye4 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye5 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye0 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组

    dataSubsampled(sampleNum)  # 依次执行功能代码
    labelShow()
    dataProcessing()
    dataShow()
