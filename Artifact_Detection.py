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
import copy
from sklearn import preprocessing
from my_utils import *

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
def data_Subsampled(orgBCG=[], orgLabel=[], sampleNum=1):
    """
    Author:Qz
    函数说明:对原时间序列进行降采样
    :param sampleNum:             输入降采样倍数
    :return:                      无
    """
    for i in range(len(orgBCG) // sampleNum):
        newBCG[i] = orgBCG[i * sampleNum]
    for i in range(orgLabel.shape[0]):
        orgLabel[i][2] //= sampleNum  # sunshi weibuzhudao
        orgLabel[i][3] //= sampleNum

    print('降采样倍数：%d' % sampleNum)
    print('原始数据长度：%d' % len(orgBCG))
    print('降采样后长度：%d' % len(newBCG))
    print('体动个数：%d' % orgLabel.shape[0])

    return newBCG, orgLabel


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
        temBCG[orgLabel[i][2]:orgLabel[i][3]] = ArtifactData_tpye0[orgLabel[i][2]:orgLabel[i][3]]


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
    '''
    重新排序后赋值并不需要深拷贝，直接赋值即可
    之前之所以会出现后面加入的新标签排序后能画出来但没有显示text文本信息，以及错位显示，是因为排序对orgLabel排序只对当前函数有效
    （其中能画出来是因为画图部分，只要给出起止位置即可画图，与text没有任何关系）
    final_figure函数里面的orgLabel依旧是没有排序的全局变量，不受排序影响
    所以才导致画图时，更新了后面补的标签的图示颜色，但没有显示对应的标签内容text，以及莫名其妙的图框外的text信息
    '''
    figure_num = [1, math.ceil(len(orgLabel) / single_check)]  # 用于显示第几画和共几画，如果能知道一个函数被调用多少次就好了
    artifact_num = 0  # 记录开始的体动xuhao，确切来说应该叫artifact_num
    while artifact_num >= 0:
        if artifact_num == 0:  # 第一画的前面不用交叠，所以单独写
            final_figure(end=orgLabel[single_check][3] + overlap, text_switch=True, artifact_num=artifact_num,
                         single_check=single_check, figure_num=figure_num, newLabel=orgLabel)
            artifact_num += single_check  # 跳转到下一个画面的开始
            figure_num[0] += 1  # 画面数加1
        elif artifact_num + single_check >= len(orgLabel):  # 最后一画最特殊，需要考量的东西较多，bug也多
            artifact_num = len(orgLabel) - single_check - 1  # 统一修改start_point，不用额外修改single_check，省事
            final_figure(start=orgLabel[-single_check][2] - overlap, text_switch=True, artifact_num=artifact_num,
                         single_check=single_check, figure_num=figure_num, newLabel=orgLabel)
            # break  # 跳出死循环
            artifact_num = -1
        else:  # 正常情况的循环
            final_figure(start=orgLabel[artifact_num][2] - overlap,
                         end=orgLabel[artifact_num + single_check][3] + overlap, text_switch=True,
                         artifact_num=artifact_num, single_check=single_check, figure_num=figure_num, newLabel=orgLabel)
            artifact_num += single_check
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


def final_figure(start=None, end=None, text_switch=False, artifact_num=0, single_check=10, figure_num=[],
                 newLabel=None):
    plt.figure(figsize=(15, 10))

    if text_switch == True:  # 开启标签序号和起止位置显示
        if start == None:  # 如果没有这一步，后面会有一些细节错误，保险一点
            start = 0
        if end == None:
            end = len(newBCG)

        for i in range(artifact_num, artifact_num + single_check):
            # if len(newBCG[newLabel[i][2]:newLabel[i][3]]) != 0:  # 有时候会出现空序列，意思是某个体动范围内有空值？无解
            plt.text((newLabel[i][2] + newLabel[i][3]) // 2 - start,  # - start是相对于当前画面的起始点而言
                     np.average(newBCG[newLabel[i][2]:newLabel[i][3]]) + 130,
                     # 用numpy的max比普通max快很多，表现在拖动图片时延迟低很多
                     (newLabel[i][0], newLabel[i][1]), fontsize=12, color="Black", style="italic", weight="light",
                     verticalalignment='center', horizontalalignment='center', rotation=0)
            plt.text((newLabel[i][2] + newLabel[i][3]) // 2 - start,  # - start是相对于当前画面的起始点而言
                     np.average(newBCG[newLabel[i][2]:newLabel[i][3]]) + 110,
                     (down_sample_rate * newLabel[i][2] / 1000, down_sample_rate * newLabel[i][3] / 1000),
                     fontsize=12, color="Black", style="italic", weight="light", verticalalignment='center',
                     horizontalalignment='center', rotation=0)
            print('average = ', np.average(newBCG[newLabel[i][2]:newLabel[i][3]]) + 110)

    plt.plot(ArtifactData_tpye1[start:end], color='red', label="大体动")
    plt.plot(ArtifactData_tpye2[start:end], color='blue', label="小体动")
    plt.plot(ArtifactData_tpye3[start:end], color='Yellow', label="深呼吸")
    plt.plot(ArtifactData_tpye4[start:end], color='purple', label="脉冲体动")
    plt.plot(ArtifactData_tpye5[start:end], color='orange', label="无效片段")
    plt.plot(ArtifactData_tpye0[start:end], color='green', label="正常数据")
    # plt.ylim(-np.average(newBCG)-200,np.average(newBCG)+200)

    '''
    可以说很离谱了，x_tick在test测试不行，在实际使用又可以
    test中显示的波形坐标还是会跟plt.xticks(x_tick)有关联，但plt.xticks(ticks=x_tick, labels=x_label)又是可以的
    而实际使用，x_tick = range(start, end, 100)却可以直接作为plt.xticks(x_tick)的参数，不会因为start的位置对波形位置产生影响
    实际使用中，plt.xticks(ticks=x_tick, labels=x_label)很难计算对应位置，总是报错不匹配
    还是会有影响，比如把end*10就出问题了，只是之前恰巧不知为啥避开了问题
    在预处理之前，已经进行过降采样，如果要显示1000Hz的坐标，必须都*10
    
    # x_tick = range(start, end, 1000 // down_sample_rate)
    # plt.xticks(x_tick)
    # 这四行程序，两者的效果是一样的
    # x_label = range(start, end, 100)
    # x_tick = range(start, end, math.ceil((end - start) / len(x_label)))

    x_label = range(start*10, end*10, 500)
    x_label = np.array(x_label).astype(str) #可要可不要
    x_tick = range(start, end, math.ceil((end - start) / len(x_label))) #start必须从0开始
    '''
    # 最终这个方法最完美，可以随意自定义
    # x_label = range(start * 10, end * 10, 500)# 1000Hz显示，内容太多了，容易重叠，不便于观察
    # x_label = list(range(start, end, 100))  #range() 返回的是“range object”，而不是实际的list 值
    # x_label = range(start, end, 100)  # range() 返回的是“range object”，而不是实际的list 值
    x_label = np.arange(start, end, 100)
    x_label = np.array(x_label) / 100  # // sunshi

    # print(start)
    # for i in range(10):
    #     print(x_label[i])

    # x_label = np.array(x_label).astype(str) #可要可不要
    # 虚惊一场，x_tick的开始必须是0，不然第二个窗口开始，又会出问题，总之ticks参数还是和波形有关联，必须和xlim一致，lebels则可以自定义
    # x_tick = range(0, end - start, math.ceil((end - start) / len(x_label)))   #取整会丢弃部分精度，会累加，导致坐标与实际产生偏差
    # x_tick = np.linspace(0, end - start, len(x_label))  # linspace比range好用，不用自己求步进值，避免长度不对
    # np.linspace是从0开始算的，好比[0:10]实际是0~9整好十个，是没有算上end的，故endpoint必须为False，否则不均匀
    # linspace比range好用，不用自己求步进值，避免长度不对
    x_tick = np.linspace(0, 100 * len(x_label), len(x_label), endpoint=False, dtype=int)
    # x_tick = range(0, 100 * len(x_label), 100)  # np.linspace不知为何会导致间隔100.15...endpoint改成False又不会了，或者用range

    # print('len(x_label) : ', len(x_label))
    # print('linspace = ', x_tick[:100])

    # for i in range(10):
    #     print(x_tick[i])
    plt.xticks(ticks=x_tick, labels=x_label)

    plt.xlabel('X Axis ：刻度为在原信号中的起始与终止位置')
    plt.ylabel('Y Axis ：Voltage')
    # plt.ylim(1100, 2600)
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


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


if __name__ == "__main__":
    filePath = "/home/qz/文档/Qz/workspace/ArtifactDataset/巢守达 2018081722-43-54"  # 文件夹的绝对路劲
    bcgPath = filePath + "/raw_org.txt"  # 原始数据路径
    labelPath = filePath + "/Artifact_a.txt"  # 体动数据路径
    orgBCG = pd.read_csv(bcgPath, header=None).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
    orgLabel = pd.read_csv(labelPath, header=None).to_numpy().reshape(-1, 4)  # 标签数据读取为numpy形式，并reshape为n行4列的数组

    down_sample_rate = 10  # 降采样倍数
    newBCG = np.full(len(orgBCG) // down_sample_rate, np.nan)  # 创建与orgBCG降采样后一样长度的空数组
    newBCG, orgLabel = data_Subsampled(orgBCG=orgBCG, orgLabel=orgLabel, sampleNum=down_sample_rate)
    temBCG = copy.deepcopy(newBCG)

    # newBCG = Butterworth(newBCG, type='bandpass', lowcut=0.1, highcut=20, order=2, Sample_org=100)
    # my_fft2(signal=newBCG, fs=100)

    ArtifactData_tpye0 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye1 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye2 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye3 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye4 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组
    ArtifactData_tpye5 = np.full(len(newBCG), np.nan)  # 创建与newBCG一样长度的空数组

    # 依次执行功能代码
    # newBCG = standardization(newBCG)
    # newBCG = preprocessing.normalize(newBCG.reshape(1,-1), axis=1, norm='max').reshape(-1)

    # labelShow(down_sample_rate=down_sample_rate)  #显示整份数据的标签
    dataProcessing()  # 将各类体动分别存储
    ArtifactData_tpye0 = temBCG

    # dataShow()    #单纯显示整份数据的标签情况
    # muti_dataShow(single_len=3600, Sample_org=1000, down_sample_rate=down_sample_rate)    #固定长度交叠显示
    artifact_check(orgLabel, down_sample_rate=down_sample_rate)  # 根据单次体动数量分窗显示，检测标签情况
