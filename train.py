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
from model.model import *
from model.LSTM_FCN import *
from my_utils import *
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import preprocessing
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
def data_preprocess_meth1(dataset_input):
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

    # test_time = time.perf_counter()
    # for i in range(len(data_class0_tem)):
    #     data_class0_tem[i] = Butterworth(data_class0_tem[i], type='bandpass', lowcut=2, highcut=15, order=2,
    #                                      Sample_org=100)        data_class0_tem[i] = Butterworth(data_class0_tem[i], type='bandpass', lowcut=2, highcut=15, order=2,
    #                                      Sample_org=100)
    # plt.plot(data_class0_tem[i], color='blue', label="正常数据")
    # plt.show()
    # print('Butterworth test_time = %2d min : %2d s' % (
    #     (time.perf_counter() - test_time) // 60, (time.perf_counter() - test_time) % 60))

    # for i in range(100):
    #     plt.plot(data_class5_tem[i], color='blue', label="正常数据")
    #     plt.show()

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

    # test_time = time.perf_counter()
    # # 去掉幅值超过一定范围的正常片段
    # threshold_value = 250
    # save_index = np.array([])
    # for i in range(tmp[0]):  # 执行速度非常慢，需要11s
    #     if np.max(data_class0_tem[i]) - np.min(data_class0_tem[i]) < threshold_value:
    #         save_index = np.append(save_index, i)
    #
    # tem_cout = data_class0_tem.shape[0]  #
    # data_class0_tem = data_class0_tem[save_index.astype(int), :]  # 不能直接int(save_index)
    # print('All of the delete frag is : ', tem_cout - data_class0_tem.shape[0])
    # print('预处理：剔除阈值过大的片段后剩余正常片段个数 : ', data_class0_tem.shape)
    # print('test_time = %2d min : %2d s' % (
    #     (time.perf_counter() - test_time) // 60, (time.perf_counter() - test_time) % 60))

    # 随机抽取，使样本均衡且乱序
    data_class0_tem = data_class0_tem[
        # np.random.randint(data_class0_tem.shape[0], size=min(tmp.values()))]  # 这里不能继续使用tmp[0]，越界，用剔除后的维度
        np.random.randint(data_class0_tem.shape[0], size=24645)]  # 二分类用这条语句 #size=5 * min(tmp.values())
    # data_class1_tem = data_class1_tem[np.random.randint(tmp[1], size=4 * min(tmp.values()))]
    # data_class2_tem = data_class2_tem[np.random.randint(tmp[2], size=min(tmp.values()))]
    # data_class3_tem = data_class3_tem[np.random.randint(tmp[3], size=3 * min(tmp.values()))]
    # data_class4_tem = data_class4_tem[np.random.randint(tmp[4], size=min(tmp.values()))]
    # data_class5_tem = data_class5_tem[np.random.randint(tmp[5], size=min(tmp.values()))]

    # preproccessed_data = np.vstack(
    prep_data_ch0 = np.vstack(
        # (data_class0_tem, data_class1_tem, data_class2_tem, data_class3_tem, data_class4_tem, data_class5_tem))
        (data_class0_tem, data_class1_tem, data_class2_tem))

    prep_data_ch1 = np.zeros_like(prep_data_ch0)
    # for i in tqdm(range(len(prep_data_ch1)), desc="prep_data_ch1 filting : "):
    #     prep_data_ch1[i] = Butterworth(prep_data_ch0[i], type='lowpass', lowcut=1, order=2, Sample_org=100)
    # for i in tqdm(range(len(prep_data_ch0)), desc="prep_data_ch1 filting : "):
    #     prep_data_ch0[i] = Butterworth(prep_data_ch0[i], type='bandpass', lowcut=2, highcut=15, order=2, Sample_org=100)

    # for i in tqdm(range(len(prep_data_ch0)), desc="prep_data_ch0 min_max normalize : "):
    #     prep_data_ch0[i] = preprocessing.MinMaxScaler().fit_transform(prep_data_ch0[i].reshape(-1, 1)).reshape(1, -1)
    # for i in tqdm(range(len(prep_data_ch1)), desc="prep_data_ch1 min_max normalize : "):
    #     prep_data_ch1[i] = preprocessing.MinMaxScaler().fit_transform(prep_data_ch1[i].reshape(-1, 1)).reshape(1, -1)

    prep_data_ch0 = prep_data_ch0.reshape(prep_data_ch0.shape[0], 1, prep_data_ch0.shape[1])
    prep_data_ch1 = prep_data_ch1.reshape(prep_data_ch1.shape[0], 1, prep_data_ch1.shape[1])
    # preproccessed_data = np.array([prep_data_ch0, prep_data_ch1])
    prep_data = np.concatenate((prep_data_ch0, prep_data_ch1), axis=1)
    print('New shape of preproccessed_data is : ', prep_data.shape)

    # for i in range(20):
    #     plt.plot(prep_data[i, 0, :], color='blue', label="CH0")
    #     plt.plot(prep_data[i, 1, :] + 1, color='red', label="CH1")
    #     plt.show()

    # preproccessed_label = np.hstack((np.full(min(tmp.values()), 0), np.full(min(tmp.values()), 1),
    #                                  np.full(min(tmp.values()), 2), np.full(min(tmp.values()), 3),
    #                                  np.full(min(tmp.values()), 4), np.full(min(tmp.values()), 5)))
    # prep_label = np.hstack((np.full(5 * min(tmp.values()), 0), np.full(5 * min(tmp.values()), 1)))  # 二分类
    prep_label = np.hstack((np.full(24645, 0), np.full(24645, 1)))  # 二分类

    return prep_data_ch0, prep_label


def data_preprocess_meth2(dataset_input, win_width = 5, time_gran = 1):
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

    # 随机抽取，使样本均衡且乱序
    data_class0_tem = data_class0_tem[ # 这里不能继续使用tmp[0]，越界，用剔除后的维度
        np.random.randint(data_class0_tem.shape[0], size=41605)]  # 二分类用这条语句 #size=5 * min(tmp.values())
    # data_class1_tem = data_class1_tem[np.random.randint(tmp[1], size=40000)]
    # data_class2_tem = data_class2_tem[np.random.randint(tmp[2], size=min(tmp.values()))]
    # data_class3_tem = data_class3_tem[np.random.randint(tmp[3], size=3 * min(tmp.values()))]
    # data_class4_tem = data_class4_tem[np.random.randint(tmp[4], size=min(tmp.values()))]
    # data_class5_tem = data_class5_tem[np.random.randint(tmp[5], size=min(tmp.values()))]

    # preproccessed_data = np.vstack(
    prep_data_ch0 = np.vstack(
        # (data_class0_tem, data_class1_tem, data_class2_tem, data_class3_tem, data_class4_tem, data_class5_tem))
        (data_class0_tem, data_class1_tem, data_class2_tem))

    prep_label    = prep_data_ch0[:, :win_width]
    prep_data_ch0 = prep_data_ch0[:, win_width:]

    prep_data_ch1 = np.zeros_like(prep_data_ch0)
    # for i in tqdm(range(len(prep_data_ch1)), desc="prep_data_ch1 filting : "):
    #     prep_data_ch1[i] = Butterworth(prep_data_ch0[i], type='lowpass', lowcut=1, order=2, Sample_org=100)
    # for i in tqdm(range(len(prep_data_ch0)), desc="prep_data_ch1 filting : "):
    #     prep_data_ch0[i] = Butterworth(prep_data_ch0[i], type='bandpass', lowcut=2, highcut=15, order=2, Sample_org=100)

    prep_data_ch0 = prep_data_ch0.reshape(prep_data_ch0.shape[0], 1, prep_data_ch0.shape[1])
    prep_data_ch1 = prep_data_ch1.reshape(prep_data_ch1.shape[0], 1, prep_data_ch1.shape[1])
    # preproccessed_data = np.array([prep_data_ch0, prep_data_ch1])
    prep_data = np.concatenate((prep_data_ch0, prep_data_ch1), axis=1)
    print('New shape of preproccessed_data is : ', prep_data.shape)

    # preproccessed_label = np.hstack((np.full(min(tmp.values()), 0), np.full(min(tmp.values()), 1),
    #                                  np.full(min(tmp.values()), 2), np.full(min(tmp.values()), 3),
    #                                  np.full(min(tmp.values()), 4), np.full(min(tmp.values()), 5)))
    # prep_label = np.hstack((np.full(5 * min(tmp.values()), 0), np.full(5 * min(tmp.values()), 1)))  # 二分类

    return prep_data_ch0, prep_label


def main():
    # Hyper Parameters
    EPOCH        = 100       # 训练整批数据多少次
    BATCH_SIZE   = 512       # 每次训练多少份数据
    split_ratio  = 0.7       # 训练集与验证集的划分
    TIME_STEP    = 1         # rnn 时间步数 / 图片高度
    INPUT_SIZE   = 500       # rnn 每步输入值 / 图片每行像素
    LR           = 0.0001    # learning rate
    thre_jud     = 0.6       # sigmoid出来后进行阈值判断，大于该值判为1
    sample_rate  = 100
    win_width    = 10        # 样本长度，单位为秒
    time_gran    = 1         # 时间粒度，单位为秒
    start        = time.perf_counter()  # Python 3.8不支持time.clock()
    random.seed(1)           # 随机数种子

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    # print('Using {0} device. | Using {1} dataloader workers every process'.format(device, nw))

    print('\033[1;32m Loading... \033[0m' % ())
    dataset_load = np.load('./dataset/dataset_meth2_Version_1.0(10s_7_sample_15Hz_lowpass).npy')  # 加载数据集，包含第一列的标签和后面的数据
    print('\033[1;32m Loading is complete, a total of time-consuming is : %f\033[0m' % (time.perf_counter() - start))

    # show_num = 64
    # for i in range(50):
    #     frag_check_multi_show(signal=dataset_load, start_point=5000, win_count=show_num)

    data_load, label_load = data_preprocess_meth2(dataset_input = dataset_load, win_width = win_width, time_gran = time_gran) # 对加载的数据集进行预处理

    data_load  = torch.tensor(data_load,  dtype=torch.float32)   # Tensor和tensor不一样，或使用torch.from_numpy()
    label_load = torch.tensor(label_load, dtype=torch.long)      # torch.tensor可以同时转tensor和dtype

    perm = torch.randperm(len(data_load))  # 返回一个0到n-1的数组0（同时随机打乱数据集和标签）
    data_load  = data_load[perm]           # 一种新的打乱方法，也可以直接用新建数组，打乱数组作为index方式
    label_load = label_load[perm]          # 这种用法必须是tensor

    print('The shape of data_load is {0} | label_load is {1}'.format(data_load.shape, label_load.shape))

    # for i in range(100):
    #     plt.figure(figsize=(16, 8))
    #     plt.plot(data_load[i,0,:])
    #     plt.title(label_load[i,:].numpy())
    #     plt.ylim(900, 3300)
    #     plt.show()


    # b = a.transpose(1, 2)  # 交换第二维度和第三维度，改变了内存的位置
    # data_load = data_load.reshape(data_load.shape[0], 1, data_load.shape[1])  # reshape成Cov1D的输入维度
    # data_load = data_load.reshape(data_load.shape[0], 1, data_load.shape[2])  # reshape成Cov1D的输入维度
    # label_load = label_load.reshape(label_load.shape[0], 1)   #之前好像需要这步操作，现在加上会报错，维度错误
    # label_load = label_load.reshape(label_load.shape[0])  #相当于没有操作

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    train_dataset    = torch.utils.data.TensorDataset(data_load[:int(split_ratio * len(data_load))], label_load[:int(split_ratio * len(label_load))])
    validate_dataset = torch.utils.data.TensorDataset(data_load[int(split_ratio * len(data_load)):], label_load[int(split_ratio * len(label_load)):])

    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)

    print("using {} fragment for training, {} fragment for validation.".format(len(train_dataset), len(validate_dataset)))

    time_steps = win_width * sample_rate
    num_variables = 2

    # model = vgg(model_name="vgg16", num_classes=2, init_weights=False)
    model = VGG(num_classes=2, init_weights=False)  #使用自己修改的CNN网络，面目全非的VGG
    # model = LSTMFCN(time_steps, num_variables)
    # model = RNN()
    # model = vgg(model_name="vgg16", num_classes=2, init_weights=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    para_num = get_parameter_number(model)  #获取模型参数量
    print('Total parameter number is : {Total:,} | Total trainable number is : {Trainable:,}'.format(Total=para_num['Total'], Trainable=para_num['Trainable'])) # {:,}  1,000,000   以逗号分隔的数字格式

    save_path = './pth/{}Net.pth'.format('LSTM-FCN')

    # loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_function = nn.BCELoss()  # BCELoss可以接收和input一样维度的target，而CrossEntropyLoss只能接收(N)的target
    # loss_function = nn.NLLLoss()  # weight=weights
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0  # 记录最佳验证精度
    label_true_tem, label_pred_tem, label_prob_tem = np.array([]), np.array([]), np.array([])
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = [], [], [], []

    for epoch in range(EPOCH):
        model.train()  # 切换到训练模式
        train_acc, train_loss, train_num = 0.0, 0.0, 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            train_datas, train_labels = data
            optimizer.zero_grad()

            train_datas, train_labels = train_datas.float().cuda(), train_labels.float().cuda()  # BCELoss()要求输入是float型

            outputs = model(train_datas.cuda())

            outputs_prob = nn.functional.sigmoid(outputs).view(outputs.size(0), -1)  # BCELoss之前需要手动加入sigmoid激活

            loss = loss_function(outputs_prob, train_labels.view(outputs.size(0), -1).cuda())

            # loss = loss_function(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            # train_bar.desc = "Epoch: {}/{} Training   loss:{:.3f}  ".format(epoch + 1, EPOCH, loss)  # 进度条前面显示的内容
            train_bar.desc = "Training   loss:{:.3f}  ".format(loss)  # 进度条前面显示的内容

            train_acc += ((outputs_prob > thre_jud) == train_labels).float().sum().cpu().data.numpy()
            train_num += train_labels.shape[0]
        train_loss_all = np.append(train_loss_all, train_loss)
        train_acc_all  = np.append(train_acc_all, train_acc / train_num * time_gran / win_width) #因为一个片段内含有十个标签，所以要除

        '''
        用测试集进行结果测试时，一定要用model.eval()把dropout关掉，因为这里目的是测试训练好
        的网络，而不是训练网络，没必要再dropout和计算BN的方差和均值(BN使用训练的历史值)。
        '''
        model.eval() # validate
        val_acc, val_loss, val_num = 0.0, 0.0, 0.0 # accumulate accurate number / epoch
        label_true_tem, label_pred_tem, label_prob_tem = 0, 0, 0
        with torch.no_grad():
            val_loss = 0.0
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_datas, val_labels = val_data

                val_datas, val_labels = val_datas.float().cuda(), val_labels.float().cuda()

                outputs = model(val_datas.cuda())

                outputs_prob = nn.functional.sigmoid(outputs).view(outputs.size(0), -1)
                loss = loss_function(outputs_prob, val_labels.view(outputs.size(0), -1))

                # predict_y = torch.max(outputs, dim=1)[1]
                label_true_tem = np.append(label_true_tem, val_labels.cpu().numpy())
                label_pred_tem = np.append(label_pred_tem, torch.max(outputs, dim=1)[1].cpu().numpy()).astype(int)
                label_prob_tem = np.append(label_prob_tem, torch.max(torch.softmax(outputs, dim=1).cpu(), dim=1)[0])

                # if epoch == 9:
                # for i in range(len(val_datas)):
                #     plt.plot(val_datas[i, 0, :], color='blue', label="CH0")
                # plt.plot(val_datas[i, 1, :] + 200, color='red', label="CH1")
                # plt.title('The [predict / real]  is [{0} / {1}]'.format(predict_y[i], val_labels[i]))
                # plt.show()

                # if epoch == 19:
                #     outputs_prob = outputs_prob.float().cpu().numpy()
                #     outputs_prob[outputs_prob >= thre_jud] = 1
                #     outputs_prob[outputs_prob < thre_jud]  = 0
                #     val_datas  = val_datas.cpu().numpy()
                #     for i in range(100):
                #         plt.figure(figsize=(16, 8))
                #         plt.plot(val_datas[i, 0, :])
                #         plt.title(outputs_prob[i, :])
                #         plt.ylim(1200, 2400)
                #         plt.show()

                # val_acc += torch.eq(predict_y, val_labels.cuda()).sum().item()

                val_loss += loss.item()
                val_acc  += ((outputs_prob > thre_jud) == val_labels).float().sum().cpu().data.numpy()
                val_num  += val_labels.shape[0]

                val_bar.desc = "Validating loss:{:.3f} ".format(val_loss)  # 进度条前面显示的内容

            val_loss_all = np.append(val_loss_all, val_loss)
            val_acc_all  = np.append(val_acc_all, val_acc / val_num * time_gran / win_width)

        val_accurate = val_acc / len(validate_dataset) * time_gran / win_width #1000是片段长度，100是每个小标签的长度，10秒里面有10个
        print('\r[epoch %d] train_loss: %.3f  val_accuracy: %.3f\r' % (epoch + 1, train_loss / len(train_loader), val_accurate))

        # statistics_show(label_true=label_true_tem, label_pred=label_pred_tem, label_prob=label_prob_tem)

        if val_accurate > best_val_acc:
            best_val_acc = val_accurate
            torch.save(model.state_dict(), save_path)
            label_true, label_pred, label_prob = label_true_tem, label_pred_tem, label_prob_tem


    print('Finished Training! The best val_accurate is : {0}'.format(best_val_acc))
    print('Total program execution time = %2d min : %2ds' % (
        (time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))

    print('The shape of label_true is {0} | label_pred is {1} | label_prob is {2}: '.format(label_true.shape,
                                                                                            label_pred.shape,
                                                                                            label_prob.shape))

    # statistics_show(label_true=label_true, label_pred=label_pred, label_prob=label_prob)

    plt.plot(train_loss_all, color='green', label='train_loss')
    plt.plot(train_acc_all, color='red', label='train_acc')
    plt.plot(val_loss_all, color='blue', label='val_loss')
    plt.plot(val_acc_all, color='Yellow', label='val_acc')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
