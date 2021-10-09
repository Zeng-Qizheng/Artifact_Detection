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
import math

# @vectorize(['complex64(complex64, complex64)'], target='gpu')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，否则图内标注无法显示中文，SimHei是黑体
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


def labelShow(down_sample_rate=10):
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
                     (down_sample_rate * orgLabel[i][2] // 1000, down_sample_rate * orgLabel[i][3] // 1000),
                     fontsize=6, color="DarkSlateGray", style="italic", weight="light", verticalalignment='center',
                     horizontalalignment='center', rotation=0)


def dataProcessing():
    """
    Author:Qz
    函数说明:将不同类型的体动分别赋值给多个一维数组
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


def artifact_check(orgLabel, Sample_org=1000, down_sample_rate=10):
    """
    Author:Qz
    函数说明:分窗检查标签情况，每个窗的体动数量可调
    :param single_check:  单画面显示的体动个数（不含交叠部分的体动）
    :return:              无
    """
    single_check = 10  # 单个画面显示十个体动（不含交叠部分的体动）
    overlap = 30 * (Sample_org // down_sample_rate)  # 交叠30s，没必要交叠太多，仅查看是否漏标或标窄了，不可能体动卡在中间

    # 很多体动是乱序的，有些是在后面补上的
    sort = np.lexsort(orgLabel.T[:3, :])  # 默认对二维数组最后一行排序并返回索引
    orgLabel = orgLabel[sort, :]  # 根据体动类型排序

    figure_num = [1, math.ceil(len(orgLabel) / single_check)]  # 用于显示第几画和共几画，如果能知道一个函数被调用多少次就好了
    start_point = 0  # 记录开始的体动个数，确切来说应该叫artifact_num
    while True:
        if start_point == 0:  # 第一画的前面不用交叠，所以单独写
            final_figure(end=orgLabel[single_check][3] + overlap, text_switch=True, start_point=start_point,
                         single_check=single_check, figure_num=figure_num)
            start_point += single_check  # 跳转到下一个画面的开始
            figure_num[0] += 1  # 画面数加1
        elif start_point + single_check >= len(orgLabel):  # 最后一画最特殊，需要考量的东西较多，bug也多
            start_point = len(orgLabel) - single_check - 1  # 统一修改start_point，不用额外修改single_check，省事
            final_figure(start=orgLabel[-single_check][2] - overlap, text_switch=True, start_point=start_point,
                         single_check=single_check, figure_num=figure_num)
            break  # 跳出死循环
        else:  # 正常情况的循环
            final_figure(start=orgLabel[start_point][2] - overlap,
                         end=orgLabel[start_point + single_check][3] + overlap, text_switch=True,
                         start_point=start_point, single_check=single_check, figure_num=figure_num)
            start_point += single_check
            figure_num[0] += 1


# @jit(nopython=True) #numba Python加速的一种装饰器，尚未调试成功
def muti_dataShow(single_len=3600, Sample_org=1000, down_sample_rate=10):
    """
    Author:Qz
    函数说明:将数据分多附图显示，但目前效果不尽人意，依旧拖动卡顿严重
    函数说明（改）:将数据分多附图显示，前后交叠一部分，但依旧不利于检查标签，因为标签部分不容易分窗显示，最好根据标签来确定显示长度
    :param showTimes:     分图个数
    :return:              无
    """
    show_len = single_len * (Sample_org // down_sample_rate)
    overlap = 600 * (Sample_org // down_sample_rate)

    start_point = 0
    while start_point + show_len < len(newBCG):
        final_figure(start=start_point, end=start_point + show_len)
        start_point += show_len - overlap

    final_figure(start=len(newBCG) - show_len)

    # for i in range(showTimes):
    #     plt.plot(ArtifactData_tpye1, color='red', label="大体动")
    #     plt.plot(ArtifactData_tpye2, color='blue', label="小体动")
    #     plt.plot(ArtifactData_tpye3, color='brown', label="深呼吸")
    #     plt.plot(ArtifactData_tpye4, color='purple', label="脉冲体动")
    #     plt.plot(ArtifactData_tpye5, color='orange', label="无效片段")
    #     plt.plot(newBCG, color='green', label="正常数据")
    #     plt.legend(ncol=2)
    #
    #     dataLen = len(newBCG)
    #     plt.xlim(dataLen * i // showTimes, dataLen * (i + 1) // showTimes)
    #     # plt.ylim(1550, 2250)
    #     # plt.savefig(filePath + "/raw_org_label.jpg", dpi=300)
    #     plt.show()


def dataShow(showTimes=1):
    """
    Author:Qz
    函数说明:单纯画图，一幅图
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


def final_figure(start=None, end=None, text_switch=False, start_point=0, single_check=10, figure_num=[]):
    plt.figure(figsize=(15, 10))

    if text_switch == True:  # 开启标签序号和起止位置显示
        if start == None:  # 如果没有这一步，后面会有一些细节错误，保险一点
            start = 0
        if end == None:
            end = len(newBCG)

        for i in range(start_point, start_point + single_check):
            # if len(newBCG[orgLabel[i][2]:orgLabel[i][3]]) != 0:  # 有时候会出现空序列，意思是某个体动范围内有空值？无解
            plt.text((orgLabel[i][2] + orgLabel[i][3]) // 2 - start,  # - start是相对于当前画面的起始点而言
                     np.average(orgBCG[orgLabel[i][2]:orgLabel[i][3]]) + 120,
                     # 用numpy的max比普通max快很多，表现在拖动图片时延迟低很多
                     orgLabel[i][0], fontsize=12, color="Black", style="italic", weight="light",
                     verticalalignment='center', horizontalalignment='center', rotation=0)
            plt.text((orgLabel[i][2] + orgLabel[i][3]) // 2 - start,  # - start是相对于当前画面的起始点而言
                     np.average(orgBCG[orgLabel[i][2]:orgLabel[i][3]]) + 110,
                     (down_sample_rate * orgLabel[i][2] // 1000, down_sample_rate * orgLabel[i][3] // 1000),
                     fontsize=12, color="Black", style="italic", weight="light", verticalalignment='center',
                     horizontalalignment='center', rotation=0)

    plt.plot(ArtifactData_tpye1[start:end], color='red', label="大体动")
    plt.plot(ArtifactData_tpye2[start:end], color='blue', label="小体动")
    plt.plot(ArtifactData_tpye3[start:end], color='Yellow', label="深呼吸")
    plt.plot(ArtifactData_tpye4[start:end], color='purple', label="脉冲体动")
    plt.plot(ArtifactData_tpye5[start:end], color='orange', label="无效片段")
    plt.plot(newBCG[start:end], color='green', label="正常数据")
    # plt.ylim(-np.average(newBCG)-200,np.average(newBCG)+200)
    plt.ylim(1550, 2150)
    plt.title('第 {0} 画 / 共 {1} 画'.format(figure_num[0], figure_num[1]), fontsize=20)
    plt.legend(ncol=2)
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
    filePath = "/home/qz/文档/Qz/workspace/ArtifactDataset/岑玉娴2018060622-16-25"  # 文件夹的绝对路劲
    bcgPath = filePath + "/raw_org.txt"  # 原始数据路径
    labelPath = filePath + "/Artifact_a.txt"  # 体动数据路径
    orgBCG = pd.read_csv(bcgPath, header=None).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
    orgLabel = pd.read_csv(labelPath, header=None).to_numpy().reshape(-1, 4)  # 标签数据读取为numpy形式，并reshape为n行4列的数组

    # orgBCG = Butterworth(orgBCG, type='bandpass', lowcut=2, highcut=5, order=2, Sample_org=1000)

    down_sample_rate = 10  # 降采样倍数
    newBCG = np.full(len(orgBCG) // down_sample_rate, np.nan)  # 创建与orgBCG降采样后一样长度的空数组
    ArtifactData_tpye1 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye2 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye3 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye4 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye5 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye0 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组

    # 依次执行功能代码
    dataSubsampled(down_sample_rate)
    # labelShow(down_sample_rate=down_sample_rate)  #显示整份数据的标签
    dataProcessing()  # 将各类体动分别存储
    # dataShow()    #单纯显示整份数据的标签情况
    # muti_dataShow(single_len=3600, Sample_org=1000, down_sample_rate=down_sample_rate)    #固定长度交叠显示
    artifact_check(orgLabel, down_sample_rate=down_sample_rate)  # 根据单次体动数量分窗显示，检测标签情况
