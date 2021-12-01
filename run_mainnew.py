# encoding:utf-8

import sys
import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import wfdb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from BCGDataset import BCGDataset,BCG_Operation,read_all_data
from Deep_Model import Unet,fcn,SegNet,DeepLabV3Plus,ResUNet,DUNet,LSTM_UNet,R2U_Net,AttU_Net,R2AttU_Net,Unet_lstm,deep_Unet,Fivelayer_Unet,Sixlayer_Unet,Threelayer_Unet
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

torch.manual_seed(31)
torch.cuda.manual_seed(31)
warnings.filterwarnings("ignore")

def read_record(filename,posture="ALL"):                  #读取数据函数，ALL为所有，输入1，2，3为某一个文件；
    posture_list = os.listdir(filename)
    #print(posture_list)
    if posture=="ALL":
        file_dir = [os.path.join(filename,posture) for posture in posture_list]
        BCG_dir = [os.path.join(dir, "orgData_down.txt") for dir in file_dir]
        ECG_dir = [os.path.join(dir, "ECG_down.txt") for dir in file_dir]
        label_dir  = [os.path.join(dir, "label2_down.txt") for dir in file_dir]
        BCG = []
        ECG = []
        label = []
        for dir1,dir2,dir3 in zip(BCG_dir,label_dir,ECG_dir):
            BCG.append(  np.array(pd.read_csv(dir1,header=None)).reshape(-1).tolist())
            label.append(np.array(pd.read_csv(dir2,header=None)).reshape(-1).tolist())
            ECG.append(  np.array(pd.read_csv(dir3,header=None)).reshape(-1).tolist())
        return BCG,label,ECG
    elif posture in posture_list:
        file_dir = os.path.join(filename,posture)

        BCG_dir = os.path.join(file_dir,"orgData_down.txt")
        ECG_dir = os.path.join(file_dir, "ECG_down.txt")
        label_dir  = os.path.join(file_dir,"label2_down.txt")

        BCG   = np.array(  pd.read_csv(BCG_dir,header=None)).reshape(-1).tolist()
        label = np.array(pd.read_csv(label_dir,header=None)).reshape(-1).tolist()
        ECG   = np.array(  pd.read_csv(ECG_dir,header=None)).reshape(-1).tolist()
        return BCG,label,ECG
    else:
        BCG_dir = os.path.join(filename, "orgData_down.txt")
        ECG_dir = os.path.join(filename, "ECG_down.txt")
        label_dir = os.path.join(filename, "label2_down.txt")

        BCG = np.array(pd.read_csv(BCG_dir, header=None)).reshape(-1).tolist()
        label = np.array(pd.read_csv(label_dir, header=None)).reshape(-1).tolist()
        ECG = np.array(pd.read_csv(ECG_dir, header=None)).reshape(-1).tolist()
        return BCG, label, ECG
        raise ValueError("The posture is wrong{}".format(posture))


def calculate_beat(y,predict,up=10):                  #通过预测计算回原来J峰的坐标 输入：y_prob,predict=ture,up*10,降采样多少就乘多少
    if predict:
        beat = (y>0.5).float().view(-1).cpu().data.numpy()
    else:
        beat = y.view(-1).float().cpu().data.numpy()
    beat_diff = np.diff(beat)          #一阶差分
    up_index = np.argwhere(beat_diff == 1).reshape(-1)
    down_index = np.argwhere(beat_diff == -1).reshape(-1)
    if len(up_index)==0:
        return [0]
    if up_index[0] > down_index[0]:
        down_index = np.delete(down_index, 0)
    if up_index[-1] > down_index[-1]:
        up_index = np.delete(up_index, -1)
    predict_J = (up_index.reshape(-1) + down_index.reshape(-1)) // 2*up
    return predict_J


def add_beat(y,th=0.3):                  #通过预测计算回原来J峰的坐标 输入：y_prob,predict=ture,up*10,降采样多少就乘多少
    """
    :param y: 预测输出值或者标签值（label）
    :param predict: ture or false
    :param up: 降采样为多少就多少
    :return: 预测的J峰位置
    """

    beat1 = np.where(y>th,1,0)
    beat_diff1 = np.diff(beat1)          #一阶差分
    add_up_index = np.argwhere(beat_diff1 == 1).reshape(-1)
    add_down_index = np.argwhere(beat_diff1 == -1).reshape(-1)
    if len(add_up_index) > 0:
        if add_up_index[0] > add_down_index[0]:
            add_down_index = np.delete(add_down_index, 0)
        if add_up_index[-1] > add_down_index[-1]:
            add_up_index = np.delete(add_up_index, -1)
        return add_up_index, add_down_index

    else:
        return 0

    # predict_J = predict_J.astype(int)



def new_calculate_beat(y,predict,th=0.5,up=10):                  #通过预测计算回原来J峰的坐标 输入：y_prob,predict=ture,up*10,降采样多少就乘多少
    """
    加上不应期算法，消除误判的峰
    :param y: 预测输出值或者标签值（label）
    :param predict: ture or false
    :param up: 降采样为多少就多少
    :return: 预测的J峰位置
    """
    if predict:
        beat = np.where(y>th,1,0)
    else:
        beat = y
    beat_diff = np.diff(beat)          #一阶差分
    up_index = np.argwhere(beat_diff == 1).reshape(-1)
    down_index = np.argwhere(beat_diff == -1).reshape(-1)
    # print(up_index,down_index)
    # print(y)

    # print(y[up_index[4]+1:down_index[4]+1])


    if len(up_index)==0:
        return [0]
    if up_index[0] > down_index[0]:
        down_index = np.delete(down_index, 0)
    if up_index[-1] > down_index[-1]:
        up_index = np.delete(up_index, -1)

    """
    加上若大于130点都没有一个心跳时，降低阈值重新判决一次，一般降到0.3就可以了；； 但是对于体动片段降低阈值可能又会造成误判，而且出现体动的话会被丢弃，间隔时间也长
    """
    print("初始：",up_index.shape,down_index.shape)
    i = 0
    lenth1 = len(up_index)
    while i < len(up_index)-1:
        if abs(up_index[i+1]-up_index[i]) > 130:
            re_prob = y[down_index[i]+2:up_index[i+1]]           #原本按正常应该是两个都+1的，但是由于Unet输出低于0.6时，把阈值调小后会在附近一两个点也变为1，会影响判断
            # print(re_prob.shape)
            beat1 = np.where(re_prob > 0.3, 1, 0)
            # print(beat1)
            if sum(beat1) != 0:
                insert_up_index,insert_down_index = add_beat(re_prob,th=0.3)
                # print(i)
                if len(insert_up_index) > 1:
                    l = i+1
                    for u,d in zip(insert_up_index,insert_down_index):
                        np.insert(up_index,l,u+down_index[i]+1)
                        np.insert(down_index,l,d+down_index[i]+1)
                        l = l+1
                elif len(insert_up_index) == 1:
                    # print(i)
                    up_index = np.insert(up_index,i+1,down_index[i]+insert_up_index+1)
                    down_index = np.insert(down_index,i+1,down_index[i]+insert_down_index+1)
                i = i + len(insert_up_index) + 1
            else:
                i = i+1
                continue
        else:
            i = i+1
    print("最终：",up_index.shape,down_index.shape)

    """
    添加不应期
    """
    new_up_index = up_index
    new_down_index = down_index
    flag = 0
    i = 0
    lenth = len(up_index)
    while i < lenth:
        if abs(up_index[i+1]-up_index[i]) < 45:
            prob_forward = y[up_index[i]+1:down_index[i]+1]
            prob_backward = y[up_index[i+1]+1:down_index[i+1]+1]

            forward_score = 0
            back_score = 0

            forward_count = down_index[i] - up_index[i]
            back_count = down_index[i+1] - up_index[i+1]

            forward_max = np.max(prob_forward)
            back_max = np.max(prob_backward)

            forward_min = np.min(prob_forward)
            back_min = np.min(prob_backward)

            forward_average = np.mean(prob_forward)
            back_average = np.mean(prob_backward)

            if forward_count > back_count:
                forward_score = forward_score + 1
            else:back_score = back_score + 1

            if forward_max > back_max:
                forward_score = forward_score + 1
            else:back_score = back_score + 1

            if forward_min < back_min:
                forward_score = forward_score + 1
            else:back_score = back_score + 1

            if forward_average > back_average:
                forward_score = forward_score + 1
            else:back_score = back_score + 1

            if forward_score >=3:
                up_index = np.delete(up_index, i+1)
                down_index = np.delete(down_index, i+1)
                flag = 1
            elif back_score >=3:
                up_index = np.delete(up_index, i)
                down_index = np.delete(down_index, i)
                flag = 1
            elif forward_score == back_score:
                if forward_average > back_average:
                    up_index = np.delete(up_index, i + 1)
                    down_index = np.delete(down_index, i + 1)
                    flag = 1
                else:
                    up_index = np.delete(up_index, i)
                    down_index = np.delete(down_index, i)
                    flag = 1
            if flag == 1:
                i = i
                flag = 0
            else: i = i+1

        else:i = i + 1

        if i > len(up_index)-2:
            break
        # elif abs(up_index[i+1]-up_index[i]) > 120:
    print("全部处理之后",up_index.shape,down_index.shape)
    predict_J = (up_index.reshape(-1) + down_index.reshape(-1)) // 2*up
    # predict_J = predict_J.astype(int)

    return predict_J



def train2(train_dataset, test_data,test_label, model, config):
    print("model:",config.model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader train and test
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    # loss and optimizer
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,betas=(config.beta1,config.beta2))      #优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    f1 = 0.0
    test_loss = 100000000000000000000
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    for epoch in range(config.num_epochs):
        scheduler.step()
        #  -------------------------------Training----------------------------------------
        model.train()
        train_loss, train_acc, train_num,test_acc = 0.0, 0.0, 0.0,0.0
        for X, y in train_loader:
            X = X.float().unsqueeze(1).to(device)
            y = y.float().to(device)

            y_hat = model(X)
            y_prob = F.sigmoid(y_hat).view(y_hat.size(0),-1)
            loss = loss_func(y_prob, y.view(y_hat.size(0),-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += ((y_prob>0.6) == y).float().sum().cpu().data.numpy()#【【【【【【【【【【【【【【【【【【【【【【【【【【【【
            train_num += y.shape[0]
        train_loss_all = np.append(train_loss_all,train_loss)
        train_acc_all = np.append(train_acc_all,train_acc / train_num * 100/1000)
        print(f"-------------------Train{config.n}---------------------")
        print("epoch:%d/%d: \n[Training]loss : %.5f,acc :%.5f " % (epoch, config.num_epochs, train_loss, train_acc / train_num * 100/1000))
        # ---------------------------------Testing------------------------------------------
        last_loss = test_loss
        model.eval()
        X = test_data.float().unsqueeze(1).to(device)
        y = test_label.float().to(device)
        with torch.no_grad():
            y_hat = model(X)
            y_prob = F.sigmoid(y_hat).view(y_hat.size(0), -1)
            loss = loss_func(y_prob, y.view(y_hat.size(0), -1))
        test_loss = loss.item()
        test_acc = ((y_prob > 0.6) == y).float().sum().cpu().data.numpy()
        test_loss_all = np.append(test_loss_all,test_loss)
        test_acc_all = np.append(test_acc_all,test_acc/y.size(0)*100/1000)
        #plt.figure()                                 #画出test信号与预测结果图
        #plt.plot((X.cpu().reshape(-1).data.numpy()-1850)/300)
        #plt.plot(y_prob.cpu().reshape(-1).data.numpy())
        #plt.show()
        predict_J = new_calculate_beat(y_prob, predict=True,th=0.6, up=1)
        true_J = calculate_beat(y, predict=False, up=1)
        TP = np.sum([1 for J_pre in predict_J if np.min(np.abs(true_J - J_pre)) <= 3])  # ture_J是一个数组，然后选择true_J - J_pre最小的那个
        FP = np.sum([1 for J_pre in predict_J if np.min(np.abs(true_J - J_pre)) > 3])
        FN = np.sum([1 for J_true in true_J if np.min(np.abs(predict_J - J_true)) > 3])
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        f1 = 2 * P * R / (P + R)
        print("[Testing] loss : %.5f,TP:%d,FP:%d,FN:%d,acc :%.5f,precision:%.5f ,F1:%.5f" % (test_loss, TP, FP, FN, P, R, f1))
        if last_loss >= test_loss:
            pd.DataFrame(y_prob.cpu().reshape(-1).data.numpy()).to_csv(config.result_path+ '/' + str(config.n)+".txt",index=False,header=None)
            torch.save(model.state_dict(), config.result_path + '/' + str(config.n) + '.pkl')
    print(train_loss_all,test_loss_all)
    print(train_acc_all,test_acc_all)
    plt.figure(1)
    plt.plot(train_loss_all,label = 'train_loss')
    plt.plot(test_loss_all,'r',label = 'test_loss')
    plt.legend(loc='best')
    plt.show()




def main(config):
    # 检测模型名字是否错误
    if config.model_type not in ['U_net','Unet_lstm','fcn','SegNet','DeepLabV3Plus','ResUNet','DUNet','LSTM_UNet','R2U_Net','AttU_Net','R2AttU_Net','deep_Unet','Fivelayer_Unet','Sixlayer_Unet','Threelayer_Unet']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return
    # 创建result文件夹
    config.result_path = os.path.join(config.result_path, config.model_type)       #把多个路径拼接起来
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # 数据库列表
    list = np.array(os.listdir(config.dataset_dir))
    train_index = [0,2,5,7,8,9,11,12,13,14,15,16,17,19]
    test_index = [1,3,4,6,10,18]
    train_list = [os.path.join(config.dataset_dir, train) for train in list[train_index]]
    test_list = [os.path.join(config.dataset_dir, test) for test in list[test_index]]
    train_data, train_label, test_data, test_label = np.array([]), np.array([]), np.array([]), np.array([])
    print(config.n, "-----", test_list)
    # if config.n < 0:
    #     continue
    # --------------------------    Load Dataset   ---------------------------------
    for train_dir in train_list:
        BCG, label, ECG = read_record(train_dir, posture="a")
        train_data = np.append(train_data, BCG)
        train_label = np.append(train_label, label)
        #train_data = np.append(train_data, np.array(sum(BCG, [])))
        #train_label = np.append(train_label, np.array(sum(label, [])))
    for test_dir in test_list:
        BCG, label, ECG = read_record(test_dir, posture="a")
        test_data = np.append(test_data, BCG)
        test_label = np.append(test_label, label)
        #test_data = np.append(test_data, np.array(sum(BCG, [])))
        #test_label = np.append(test_label, np.array(sum(label, [])))
    pd.DataFrame(test_data.reshape(-1)).to_csv(config.result_path + '/' + str(test_data) + ".txt",index=False, header=None)
    pd.DataFrame(test_label.reshape(-1)).to_csv(config.result_path + '/' + str(test_label) + ".txt",index=False, header=None)
    train_data = torch.tensor(train_data.reshape(-1,1000), dtype=torch.float)
    train_label = torch.tensor(train_label.reshape(-1,1000), dtype=torch.long)
    train_set = TensorDataset(train_data, train_label)
    test_data = torch.tensor(test_data.reshape(-1,1000), dtype=torch.float)
    test_label = torch.tensor(test_label.reshape(-1,1000), dtype=torch.long)

    # --------------------       Load model      ------------------------
    if config.model_type=='U_net':
        model = Unet()
    elif config.model_type=='fcn':
        model = fcn()
    elif config.model_type=='SegNet':
        model = SegNet()
    elif config.model_type=='DeepLabV3Plus':
        model = DeepLabV3Plus(n_classes=1,n_blocks=[3, 4, 4, 3],atrous_rates=[6, 12, 18],multi_grids=[1, 2, 4],output_stride=16,)
    elif config.model_type=='ResUNet':
        model = ResUNet()
    elif config.model_type=='DUNet':
        model = DUNet()
    elif config.model_type=='LSTM_UNet':
        model = LSTM_UNet()
    elif config.model_type=='R2U_Net':
        model = R2U_Net(t=config.t)
    elif config.model_type=='AttU_Net':
        model = AttU_Net()
    elif config.model_type=='R2AttU_Net':
        model = R2AttU_Net(t=config.t)
    elif config.model_type == 'Unet_lstm':
        model = Unet_lstm()
    elif config.model_type == 'deep_Unet':
        model = deep_Unet()
    elif config.model_type == 'Threelayer_Unet':    #,Fivelayer_Unet,Sixlayer_Unet,Threelayer_Unet
        model = Threelayer_Unet()
    elif config.model_type == 'Fivelayer_Unet':
        model = Fivelayer_Unet()
    elif config.model_type == 'Sixlayer_Unet':
        model = Sixlayer_Unet()
    else:
        raise ValueError("ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net")
    # ---------------------------  train  -----------------------------------
    train2(train_set, test_data, test_label, model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset_dir',type=str, default='./in_data/')
    # model hyper-parameters
    parser.add_argument('--n',type=int,default=111)
    parser.add_argument('--Signal_length', type=int, default=10000)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # momentum2 in Adam

    parser.add_argument('--model_type', type=str, default='deep_Unet', help='U_net/R2U_Net/AttU_Net/R2AttU_Net/deep_Unet')
    parser.add_argument('--result_path', type=str, default='./result/')


    config = parser.parse_args()         #解析参数，使config拥有这些设定的值
    main(config)
