import os
import json
import random
from scipy import signal
from scipy import fftpack
import numba as nb
import numpy as np
import cupy as cp
import torch
import datetime
from os import times
import time
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from model import vgg
from my_utils import *

np.set_printoptions(threshold=np.inf)  # 解决print打印太多东西自动省略的问题
np.set_printoptions(suppress=True)  # print不以科学计数法输出
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，否则图内标注无法显示中文
plt.rcParams['axes.unicode_minus'] = False

label_map = {0: "正常信号", 1: "大体动", 2: "小体动", 3: "深呼吸", 4: "脉冲体动", 5: "无效片段"}


def label_statistic(org_label, num_classes=6):  # 这个函数有点多余，写法没下面的好
    label_count = np.zeros([num_classes])
    for i in range(len(org_label)):  # 分割前的体动统计
        for t in range(0, num_classes):  # 对某个体动的类型进行逐一判断
            if org_label[i] == t:
                label_count[t] += 1

    print('Total artifact of input data is %d | Distribution of various types is ' % (np.sum(label_count[:])),
          label_count[:])


def Processing_func(input, index):  # 用于多进程调用
    if np.max(input) - np.min(input) < 150:
        save_index = np.append(save_index, index)
    print("now:" + str(datetime.datetime.now()))  # 计算算法运行时间)  # 计算算法运行时间
    return save_index


# @nb.jit()
def data_preprocess(dataset_input):
    # 首先统计输入数据集的体动各类别数量和分布情况
    mask = np.unique(dataset_input[:, 0])  # np.unique寻找数组里面的重复数值
    tmp = {}  # 字典，记录每个体动类型的个数
    for v in mask:
        tmp[v] = np.sum(dataset_input[:, 0] == v)  # 很秀的写法
    # print('The tmp is : ', tmp)  # 统计结果

    print('The label_map is : ', label_map)
    label_statistic(org_label=dataset_input[:, 0], num_classes=6)  # 统计原始数据的分布情况

    sort = np.lexsort(dataset_input.T[:1, :])  # 默认对二维数组最后一行排序并返回索引
    dataset_input = dataset_input[sort, :]  # 根据体动类型排序

    data_load = dataset_input[:, 1:]  # 把数据部分单独取出来，方便后面操作

    # 从排序好的数组分别把每一类体动单片段独取出来
    data_class0_tem = data_load[:tmp[0]]
    data_class1_tem = data_load[tmp[0]:tmp[0] + tmp[1]]
    data_class2_tem = data_load[tmp[0] + tmp[1]:tmp[0] + tmp[1] + tmp[2]]
    data_class3_tem = data_load[-tmp[3] - tmp[4] - tmp[5]:-tmp[4] - tmp[5]]
    data_class4_tem = data_load[-tmp[4] - tmp[5]:-tmp[5]]
    data_class5_tem = data_load[-tmp[5]:]

    test_time = time.perf_counter()
    for i in range(len(data_class0_tem)):
        data_class0_tem[i] = Butterworth(data_class0_tem[i], type='bandpass', lowcut=2, highcut=15, order=2,
                                         Sample_org=100)
        # plt.plot(data_class0_tem[i], color='blue', label="正常数据")
        # plt.show()
    print('Butterworth test_time = %2d min : %2d s' % (
        (time.perf_counter() - test_time) // 60, (time.perf_counter() - test_time) % 60))

    #
    # for i in range(100):
    #     plt.plot(data_class0_tem[i], color='blue', label="正常数据")
    #     plt.show()

    # if tem_flag == 0:
    #     tem_class0 = data_class0_tem[i]
    #     tem_flag += 1
    # else:
    #     tem_class0 = np.vstack((tem_class0, data_class0_tem[i]))    #这种写法运算量非常大

    # #多进程反而变慢了（11s->3s），可能是因为创建进程也消耗时间，加速的那点时间赶不上自身消耗的时间
    # pool = multiprocessing.Pool(processes=20)  # 创建20个进程
    # AllBeat_dataset = []
    # # ---------------------------按分窗进行处理--------------------------------------
    # start_time = (datetime.datetime.now())
    # for i in range(tmp[0]):
    #     input_data_pieces = data_class0_tem[i]
    #     # print("input_data_pieces:",input_data_pieces)
    #     return_data_objet = pool.apply_async(Processing_func, (input_data_pieces,i))  # 返回值得到的是一个任务的返回结果对象,而不是数值
    #     AllBeat_dataset.append(return_data_objet)
    #
    # pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join前调用
    # pool.join()  # 等待进程池中所有进程执行完毕
    # print("所有进程执行完毕")
    # end_time = (datetime.datetime.now())
    # print('Running time: %s Seconds' % (end_time - start_time))  # 计算算法运行时间
    # # ------------------总体合成区-----------------------------#
    # for win in range(len(AllBeat_dataset)):
    #     temp = AllBeat_dataset[win].get()
    #     save_index = np.append(save_index, temp)  # 根据返回对象提取对应的返回数值

    test_time = time.perf_counter()
    # 去掉幅值超过一定范围的正常片段
    threshold_value = 300
    save_index = np.array([])
    for i in range(tmp[0]):  # 执行速度非常慢，需要11s
        if np.max(data_class0_tem[i]) - np.min(data_class0_tem[i]) < threshold_value:
            save_index = np.append(save_index, i)

    tem_cout = data_class0_tem.shape[0]  #
    data_class0_tem = data_class0_tem[save_index.astype(int), :]  # 不能直接int(save_index)
    print('All of the delete frag is : ', tem_cout - data_class0_tem.shape[0])
    print('预处理：剔除阈值过大的片段后剩余正常片段个数 : ', data_class0_tem.shape)
    print('test_time = %2d min : %2d s' % (
        (time.perf_counter() - test_time) // 60, (time.perf_counter() - test_time) % 60))

    # 随机抽取，使样本均衡且乱序
    data_class0_tem = data_class0_tem[
        # np.random.randint(data_class0_tem.shape[0], size=min(tmp.values()))]  # 这里不能继续使用tmp[0]，越界，用剔除后的维度
        np.random.randint(data_class0_tem.shape[0], size=10 * min(tmp.values()))]  # 二分类用这条语句
    data_class1_tem = data_class1_tem[np.random.randint(tmp[1], size=8 * min(tmp.values()))]
    # data_class2_tem = data_class2_tem[np.random.randint(tmp[2], size=min(tmp.values()))]
    # data_class3_tem = data_class3_tem[np.random.randint(tmp[3], size=3 * min(tmp.values()))]
    data_class4_tem = data_class4_tem[np.random.randint(tmp[4], size=min(tmp.values()))]
    # data_class5_tem = data_class5_tem[np.random.randint(tmp[5], size=min(tmp.values()))]

    preproccessed_data = np.vstack(
        # (data_class0_tem, data_class1_tem, data_class2_tem, data_class3_tem, data_class4_tem, data_class5_tem))
        (data_class0_tem, data_class1_tem, data_class4_tem))
    # preproccessed_label = np.hstack((np.full(min(tmp.values()), 0), np.full(min(tmp.values()), 1),
    #                                  np.full(min(tmp.values()), 2), np.full(min(tmp.values()), 3),
    #                                  np.full(min(tmp.values()), 4), np.full(min(tmp.values()), 5)))
    preproccessed_label = np.hstack((np.full(10 * min(tmp.values()), 0), np.full(10 * min(tmp.values()), 1)))  # 二分类

    # filter_show = data_class0_tem
    # for i in range(len(filter_show)):
    #     filter_show[i] = Butterworth(filter_show[i], type='bandpass', lowcut=2, highcut=8, order=2, Sample_org=100)
    # for i in range(100):
    #     plt.plot(filter_show[i], color='blue', label="滤波后波形")
    #     plt.show()

    ###
    for i in range(len(preproccessed_data)):
        preproccessed_data[i] = Butterworth(preproccessed_data[i], type='bandpass', lowcut=2, highcut=15, order=2,
                                            Sample_org=100)

    return preproccessed_data, preproccessed_label


def main():
    # 超参数
    epochs = 10
    batch_size = 128
    split_ratio = 0.7
    random.seed(1)
    start = time.perf_counter()  # Python 3.8不支持time.clock()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print('\033[1;32m Loading... \033[0m' % ())

    dataset_load = np.load('dataset_Version_1.0.npy')  # 加载数据集，包含第一列的标签和后面的数据
    print('\033[1;32m Loading is complete, a total of time-consuming is : %f\033[0m' % (time.perf_counter() - start))

    data_load, label_load = data_preprocess(dataset_load)  # 对加载的数据集进行预处理

    # data_load = torch.from_numpy(data_load) #numpy转tensor
    # label_load = torch.from_numpy(label_load)
    data_load = torch.tensor(data_load, dtype=torch.float32)  # Tensor和tensor不一样
    label_load = torch.tensor(label_load, dtype=torch.long)  # torch.tensor可以同时转tensor和dtype

    perm = torch.randperm(len(data_load))  # 返回一个0到n-1的数组
    data_load = data_load[perm]  # 一种新的打乱方法，也可以直接用新建数组，打乱数组作为index方式
    label_load = label_load[perm]  # 这种用法必须是tensor

    # for i in range(len(label_load)):  # 将六分类变成二分类
    #     if label_load[i] > 0:
    #         label_load[i] = 1  # 二分类只有0、1，不能出现2

    print('The shape of data_load is  :', data_load.shape)
    print('The shape of label_load is :', label_load.shape)

    data_load = data_load.reshape(data_load.shape[0], 1, data_load.shape[1])  # reshape成Cov1D的输入维度
    print('The shape of data_load after reshape is :', data_load.shape)

    # label_load = label_load.reshape(label_load.shape[0], 1)   #之前好像需要这步操作，现在加上会报错，维度错误
    # label_load = label_load.reshape(label_load.shape[0])  #相当于没有操作
    print('The shape of data_load after reshape is :', label_load.shape)

    train_dataset = torch.utils.data.TensorDataset(data_load[:int(split_ratio * len(data_load))],
                                                   label_load[:int(split_ratio * len(label_load))])
    validate_dataset = torch.utils.data.TensorDataset(data_load[int(split_ratio * len(data_load)):],
                                                      label_load[int(split_ratio * len(label_load)):])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print(
        "using {} fragment for training, {} fragment for validation.".format(len(train_dataset), len(validate_dataset)))

    net = vgg(model_name="vgg16", num_classes=2, init_weights=False).to(device)
    save_path = './{}Net.pth'.format('VGG')

    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0  # 记录最佳精度
    for epoch in range(epochs):
        # train
        net.train()  # 切换到训练模式
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            datas, labels = data
            optimizer.zero_grad()
            outputs = net(datas.to(device))
            # print('The shape of outputs is :', outputs.shape)
            # print('The shape of labels is :', labels.shape)
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        '''
        用测试集进行结果测试时，一定要用net.eval()把dropout关掉，因为这里目的是测试训练好
        的网络，而不是训练网络，没必要再dropout和计算BN的方差和均值(BN使用训练的历史值)。
        '''
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_datas, val_labels = val_data
                outputs = net(val_datas.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                if epoch == 9:
                    for i in range(len(val_datas)):
                        plt.plot(val_datas[i, 0, :])
                        plt.title('The [predict / real]  is [{0} / {1}]'.format(predict_y[i], val_labels[i]))
                        plt.show()

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(validate_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(train_loader), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('The best val_accurate is : {}'.format(best_acc))

    print('Finished Training')
    print('Total program execution time = %2d min : %2ds' % (
        (time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))


if __name__ == '__main__':
    main()
