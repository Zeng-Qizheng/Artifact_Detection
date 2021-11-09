# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/25
@Auth ： 曾启正 Keyon Tsang
@File ： preproccess.py
@IDE  ： PyCharm
@Motto： ABC(Always Be Coding)
@Func ： undetermined
"""

# import numba as nb
import os
import time
from my_utils import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)  # 使print不以科学计数法输出
np.set_printoptions(threshold=np.inf)  # with open保存大量数据会出现省略号，需要用这条语句，np.savetxt则不用


# @vectorize(['complex64(complex64, complex64)'], target='gpu')


class preproccess(object):
    def __init__(self, file_path=None, sample_rate=1000, downsampling=1, artifact_stride=1, norm_stride=1, win_width=5):
        if os.path.exists(os.path.join(file_path, "raw_org.txt")) and os.path.exists(
                os.path.join(file_path, "Artifact_a.txt")):
            data_path = os.path.join(file_path, "raw_org.txt")  # 原始数据路径,"\\raw_org.txt"也可以
            label_path = os.path.join(file_path, "Artifact_a.txt")  # 体动数据路径
        elif os.path.exists(os.path.join(file_path, "new_org_1000hz.txt")) and os.path.exists(
                os.path.join(file_path, "Artifact_a.txt")):
            data_path = os.path.join(file_path, "new_org_1000hz.txt")  # 原始数据路径,"\\raw_org.txt"也可以
            label_path = os.path.join(file_path, "Artifact_a.txt")  # 体动数据路径
        else:
            print('数据或标签打开错误，不存在！')

        self.bcg_data = pd.read_csv(data_path, header=None, dtype=int).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
        self.reshape = pd.read_csv(label_path, header=None, dtype=int).to_numpy().reshape(-1, 4)
        self.bcg_label = self.reshape  # 标签数据读取为numpy形式，并reshape为n行4列的数组
        self.sample_rate = sample_rate // downsampling
        self.artifact_time_granularity = self.sample_rate * artifact_stride
        self.norm_time_granularity = self.sample_rate * norm_stride
        self.win_width = self.sample_rate * win_width
        self.downsampling = downsampling
        self.norm_data_buf = np.zeros([100000, self.win_width])  # 分割后的正常信号片段数据
        self.frag_data_buf = np.zeros([100000, self.win_width])  # 分割后的体动片段数据
        self.frag_label_buf = np.full([2, 100000], np.nan)  # 这两个的写法不太好，后期再改进
        self.all_frag_label_buf = np.array([])  # 最后将体动分割片段和正常信号分割片段的标签拼接汇总
        self.frag_count = 0  # 分割的体动片段个数
        self.norm_count = 0  # 分割的正常片段个数
        self.artifact_color = {1: "red", 2: "blue", 3: "brown", 4: "purple", 5: "orange", }
        self.label_map = {"正常信号": 0, "大体动": 1, "小体动": 2, "深呼吸": 3, "脉冲体动": 4, "无效片段": 5}
        self.label_map = dict(
            (val, key) for key, val in self.label_map.items())  # 将获得的索引和键值反过来，方便后面通过索引直接获得键值,不这样做也可以

    def one_data_split(self):
        if self.downsampling > 1:  # 默认不开启降采样
            self.data_down_sampling()
        self.reArrange()  # 分割前先对数据进行重新排列，因为打标有些前面漏的会在后面补上，

        # Butterworth Filter，统一输出15Hz低通信号
        self.bcg_data = Butterworth(self.bcg_data, type='lowpass', lowcut=15, order=2, Sample_org=self.sample_rate)

        # 体动分割
        largen_value = int(0.2 * self.win_width)    #
        for i in range(len(self.bcg_label)):  # 对体动进行片段分割
            start_point = self.bcg_label[i][2] if self.bcg_label[i][2] - largen_value < 0 else \
                self.bcg_label[i][2] - largen_value  # 三元表达式，防止开头越界；整体前后都加2.5s
            while start_point < self.bcg_label[i][3] + largen_value and (
                    start_point + self.win_width) < len(self.bcg_data):
                self.frag_data_buf[self.frag_count][:] = self.bcg_data[start_point: start_point + self.win_width]
                # tem_frag = bcg_data[start_point: start_point + win_width]
                # frag_data_buf = np.row_stack((frag_data_buf, tem_frag)) #这种写法速度非常慢
                self.frag_label_buf[0][self.frag_count] = self.bcg_label[i][0]  # 标记该片段属于哪个体动
                self.frag_label_buf[1][self.frag_count] = self.bcg_label[i][1]  # 标记该片段属于什么类型
                self.frag_count += 1
                if (start_point + self.win_width) >= self.bcg_label[i][3]:  # 判断滑窗右端是否已经超出体动范围
                    break
                start_point += self.artifact_time_granularity

        # 正常信号分割
        start_point = seq_num = 0
        while start_point < len(self.bcg_data):
            if start_point + self.win_width < self.bcg_label[seq_num][2] - int(2 * self.win_width):    #往后放宽半个
                if (np.max(self.bcg_data[start_point: start_point + self.win_width]) - np.min(
                        self.bcg_data[start_point: start_point + self.win_width]) < 1.2 * (np.max(
                        self.bcg_data[start_point - 2*self.win_width: start_point]) - np.min(
                        self.bcg_data[start_point - 2*self.win_width: start_point]))) and (np.max(
                        self.bcg_data[start_point: start_point + self.win_width]) - np.min(
                        self.bcg_data[start_point: start_point + self.win_width]) < 1.2 * (np.max(
                        self.bcg_data[start_point + self.win_width: start_point + 3 * self.win_width]) - np.min(
                        self.bcg_data[start_point + self.win_width: start_point + 3 * self.win_width]))):

                    self.norm_data_buf[self.norm_count][:] = self.bcg_data[start_point: start_point + self.win_width]
                    self.norm_count += 1
                start_point += self.norm_time_granularity
            elif seq_num < len(self.bcg_label) - 1:  # 防止越界,len返回的是个数，而seq_num是数组下标，所以不仅要用<，还要在len - 1
                start_point = self.bcg_label[seq_num][3] + int(2 * self.win_width)  #往前放宽2个
                seq_num += 1
            else:
                break

        self.all_frag_label_buf = np.hstack(  # 这里返回的标签，已经是一维数组
            (self.frag_label_buf[1][:self.frag_count], np.zeros(self.norm_count)))  # 各分割片段标签拼接汇总，体动在前正常在后

        self.org_label_statistic(self.bcg_label, num_classes=6)  # 分割完之后对体动进行统计
        self.frag_label_statistic(self.all_frag_label_buf, num_classes=6)  # 分割完之后对体动进行统计

        return np.vstack(  # 体动片段和标签在前，正常的在后
            (self.frag_data_buf[:self.frag_count][:], self.norm_data_buf[:self.norm_count][:])), self.all_frag_label_buf

    def multi_show(self, artifact_seq_num=0):  # show_data既可以加[]也可以不加
        frag_count_seq = []  # 该体动被分成的片段个数
        for index, nums in enumerate(self.frag_label_buf[0]):
            if nums == artifact_seq_num:
                frag_count_seq.append(index)  # 把属于同一个体动的分割后样本的索引，放在一个list里面
        print('The len of this Artifact is : ',
              (self.bcg_label[artifact_seq_num - 1][3] - self.bcg_label[artifact_seq_num - 1][2]) / self.sample_rate,
              's')
        print('The fragment count of this Artifact is :', len(frag_count_seq))
        win_count = len(frag_count_seq)  # 属于同一个体动的分割片段数量，每个片段作为一个窗显示
        if win_count > 99:  # 限制窗口数量，防止太长的无效或大体动片段，导致卡死
            win_count = 99
        begin_loc = frag_count_seq[0]
        line = int(win_count ** 0.5) if (int(win_count ** 0.5)) ** 2 == win_count else int(win_count ** 0.5) + 1
        list = int(win_count ** 0.5) if (int(win_count ** 0.5)) * line >= win_count else int(win_count ** 0.5) + 1

        plt.figure(figsize=(20, 10))
        plt.subplot(line + 1, 1, 1)  # 整体加多一行，使原体动片段显示在最上面一整行
        plt.plot(self.bcg_data[self.bcg_label[artifact_seq_num - 1][2]:self.bcg_label[artifact_seq_num - 1][3]],
                 # 务必务必务必记得-1
                 self.artifact_color[self.bcg_label[artifact_seq_num - 1][1]],
                 label=self.label_map[self.bcg_label[artifact_seq_num - 1][1]])  # 索引从0开始
        plt.legend(ncol=2)
        for i in range(1, win_count + 1):
            plt.subplot(line + 1, list, list + i)  # 整体加多一行，list+i是因为第一整行占了全部列，所以下一个子图在此基础上+1开始算
            # plt.ylim(0, 0.02)
            plt.plot(self.frag_data_buf[begin_loc], self.artifact_color[self.bcg_label[artifact_seq_num - 1][1]])
            begin_loc += 1
        plt.show()

    # def multi_show(show_data=[], win_count=1, begin_loc=0):
    #     global frag_label_buf, label_map, frag_count
    #     frag_color = {1: "red", 2: "blue", 3: "brown", 4: "purple", 5: "orange", }
    #     line = int(win_count ** 0.5) if (int(win_count ** 0.5)) ** 2 == win_count else int(win_count ** 0.5) + 1
    #     list = int(win_count ** 0.5) if (int(win_count ** 0.5)) * line >= win_count else int(win_count ** 0.5) + 1
    #     for i in range(1, win_count + 1):
    #         plt.subplot(line, list, i)
    #         # plt.ylim(0, 0.02)
    #         plt.plot(show_data[begin_loc], frag_color[frag_label_buf[begin_loc]], label=label_map[frag_label_buf[begin_loc]])
    #         plt.legend(ncol=2)
    #         begin_loc += 1
    #     plt.show()

    # with open("test.txt","w") as f:
    #     f.write("这是个测试！")  # 自带文件关闭功能，不需要再写f.close()
    def reArrange(self):
        print('the shape of bcg_data is :', self.bcg_data.shape)
        print('the shape of bcg_label is:', self.bcg_label.shape)

        for i in range(len(self.bcg_label)):
            if self.bcg_label[i][2] > self.bcg_label[i][3]:  # 首先判断是否有起始位置大于结束位置的情况，有则调转过来
                self.bcg_label[i][2], self.bcg_label[i][3] = self.bcg_label[i][3], self.bcg_label[i][2]

        label_index = np.lexsort(self.bcg_label.T[:3, :])  # 二维数组排序，用了lexsort方法，具体功能见OneNote
        self.bcg_label = self.bcg_label[label_index, :]
        self.bcg_label[:, 0] = range(1, len(self.bcg_label) + 1)  # 二维数组重新排序后序号也跟着移动，所以序号需要重新排列，重新赋值，从1开始

    def data_down_sampling(self):
        """
        Author:Qz
        函数说明:对原时间序列进行降采样,不用降采样是最快的，3s多一点跑完程序，十倍降采样变成3.9s，开启且只有1倍降采样要12s多
        :param sampleNum:             输入降采样倍数
        :return:                      无
        """
        temBCG = np.full(len(self.bcg_data) // self.downsampling, np.nan)  # 创建与orgBCG降采样后一样长度的空数组
        for i in range(len(self.bcg_data) // self.downsampling):
            temBCG[i] = self.bcg_data[i * self.downsampling]
        for i in range(self.bcg_label.shape[0]):
            self.bcg_label[i][2] //= self.downsampling
            self.bcg_label[i][3] //= self.downsampling

        print('已开启降采样，当前降采样倍数===>>> %d 倍' % self.downsampling)
        print('原始数据长度：%d' % len(self.bcg_data))
        self.bcg_data = temBCG
        print('降采样后长度：%d' % len(self.bcg_data))

    def org_label_statistic(self, org_label, num_classes=6):
        label_count = np.zeros([num_classes])
        for i in range(len(org_label)):  # 分割前的体动统计
            for t in range(0, num_classes):  # 对某个体动的类型进行逐一判断
                if org_label[i][1] == t:  # 这种写法没有train那种高级
                    label_count[t] += 1

        print('Total artifact of original bcg_data = %d | Distribution of various types = ' % (np.sum(label_count[:])),
              label_count[:])

    def frag_label_statistic(self, frag_label, num_classes=6):
        label_count = np.zeros([num_classes])
        for i in range(len(frag_label)):  # 分割后的体动片段类型统计
            for t in range(0, num_classes):  # 对某个体动片段的类型进行逐一判断
                if frag_label[i] == t:
                    label_count[t] += 1

        print('Total normal fragment = %d | artifact fragment = %d | Distribution of various types = ' % (
            label_count[0], np.sum(label_count[1:])), label_count[:])


def multi_data_split(filepath, file_nums='all'):
    print('该目录下共有 %d 个文件' % len(os.listdir(filepath)))

    processed_count = 0
    for i in os.listdir(filepath):  # 获取该目录下所有文件名，一个一个读取
        if os.path.isdir(os.path.join(filepath, i)):  # 判断该文件是不是目录，防止把压缩包等文件也读取进去
            processed_count += 1
            print('\033[1;32m Processing the %d / %d file, filename is : %s\033[0m' % (
                processed_count, len(os.listdir(filepath)), i))

            '''
            分割的超参在这里！！！
            '''
            data_segmentation = preproccess(file_path=os.path.join(filepath, i), downsampling=10, artifact_stride=1,
                                            norm_stride=5, win_width=5)  # 先实例化一个类，再调用函数
            # frag_dataset, frag_label = np.zeros([data_segmentation.frag_count, data_segmentation.win_width]), np.full(
            #     data_segmentation.frag_count, np.nan) # 可以不用先创建空数组，下面直接赋值即可
            frag_dataset, frag_label = data_segmentation.one_data_split()

            if processed_count == 1:
                # all_data, all_label = np.zeros([data_segmentation.frag_count, data_segmentation.win_width]), np.full(
                #     data_segmentation.frag_count, np.nan) # 这里也是可以不用先创建空数组的
                all_data, all_label = frag_dataset, frag_label
            else:
                all_data, all_label = np.vstack((all_data, frag_dataset)), np.hstack((all_label, frag_label))

            print('The shape of frag_dataset is {0} | frag_label is {1}'.format(frag_dataset.shape,frag_label.shape))

        if file_nums == 'all':
            pass
        elif processed_count == file_nums:  # 只测试十份
            break

    print('\033[1;31m 即将处理完毕 \033[0m')
    data_segmentation.frag_label_statistic(all_label, num_classes=6)

    print('The shape of finally dataset is : {0} | label is : {1}'.format(all_data.shape, all_label.shape))

    print('The label dictionary is :', data_segmentation.label_map)
    print('\033[1;31m 正在执行保存程序 \033[0m')

    # 把标签作为一列放在最左边，信号在右边，保存为(-1,501)的.npy数据文件
    final_dataset = np.hstack((all_label.reshape(-1, 1), all_data))  # all_label一维数组是横向的，先转成列数组
    print('The shape of final dataset is : ', final_dataset.shape)

    '''
    savetxt写为.npy和.txt大小没区别，而且写成的.npy无法读取，用save写入的.npy可以直接用load读取，速度极快
    save没有fmt参数，savetxt才有，报错
    用save保存为.npy速度比savetxt保存.txt快很多，25s->15s，其中12s为分割时间，实际提速4-5倍
    save写入用load读取（可不用allow_pickle=True参数），savetxt写入用loadtxt读取
    loadtxt:30.8s  load:0.15s 提速205倍
    '''
    # np.savetxt(os.path.join(os.getcwd(), "dataset","dataset.txt"), all_data,fmt = '%d')  # 保存为整数,不能写成\dataset.txt，\相当于返回根目录
    # np.savetxt(os.path.join(os.getcwd(), "dataset","label.txt"), all_label,fmt = '%d')  # 保存为整数
    np.save(os.path.join(os.getcwd(), "dataset_Version_1.3(3s_7_sample_15Hz_lowpass).npy"),
            final_dataset)  # 保存为整数,不能写成\dataset.txt，\相当于返回根目录
    # np.save(os.path.join(os.getcwd(), "label__Version_1.0.npy"), all_label)  # 保存为整数

    # with open(os.path.join(os.getcwd(), "dataset","dataset.txt"), 'w', encoding='utf-8') as file:
    #     file.write(str(all_data))   #这种方法占用内存极高，一直在上升
    #     # for i in range(len(all_data)):
    #         # file.write(str(all_data[i][:]) + "\n")    #逐行写入，这种写入方法速度极慢
    # with open(os.path.join(os.getcwd(), "dataset","label.txt"), 'w', encoding='utf-8') as file:
    #     file.write(str(all_label))
    #     # for i in range(len(all_label)):
    #     #     file.write(str(all_label[i]) + "\n")


if __name__ == "__main__":
    start = time.perf_counter()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，否则图内标注无法显示中文
    plt.rcParams['axes.unicode_minus'] = False

    # 另一种文件读取方法，各有千秋
    # with open(data_path, encoding='utf-8-sig') as file: #仅仅是读取file的地址速度快，把数据取出来很慢，23s左右
    #     org_bcg  = csv.reader(file)
    #     bcg_data = []
    #     # @nb.jit()
    #     for row in org_bcg:
    #         tem_bcg = int(row[0])
    #         bcg_data.append(tem_bcg)
    # bcg_data = np.array(bcg_data).reshape(-1)
    # print(bcg_data.reshape(-1))
    #     bcg_data = list(org_bcg)      #转成list很慢， 40s左右

    '''
    这部分是多文件分割代码入口
    '''
    file_path = '/home/qz/文档/Qz/workspace/Checked_Dataset'
    multi_data_split(file_path, file_nums='all')

    '''
    这部分是输入体动序号显示对应分割的各个片段入口代码
    '''
    # file_path = '/home/qz/文档/Qz/WorkSpace/ArtifactDataset/吴自宁20180820 22-05-16'
    # data_segmentation = preproccess(file_path, downsampling=1)  # 先实例化一个类，再调用函数
    # frag_dataset_show, frag_label_show = np.zeros([data_segmentation.frag_count, data_segmentation.win_width]), np.full(
    #     data_segmentation.frag_count, np.nan)
    # _, _ = data_segmentation.one_data_split()
    # for i in range(100, 110):
    #     data_segmentation.multi_show(i)

    '''
    这部分是正常信号分割后粗略检查的显示代码入口
    '''
    # file_path = 'E:\Qz\临床原始数据\ArtifactDataset\吴自宁20180820 22-05-16'
    # data_segmentation = preproccess(file_path, downsampling=1)  # 先实例化一个类，再调用函数
    # _, _ = data_segmentation.one_data_split()
    # for i in range(10, 110):
    #     plt.plot(data_segmentation.norm_data_buf[i][:])
    #     plt.show()

    print('The program is executed!')
    print('Total program execution time = %2d min : %2ds' % (
        (time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))
