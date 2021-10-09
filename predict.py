import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from my_utils import *

from model import vgg

np.set_printoptions(threshold=np.inf)  # 解决print打印太多东西自动省略的问题
np.set_printoptions(suppress=True)  # print不以科学计数法输出
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，否则图内标注无法显示中文
plt.rcParams['axes.unicode_minus'] = False


def main():
    batch_size = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()  # Python 3.8不支持time.clock()

    print('\033[1;32m Loading test file... \033[0m')

    file_path = '/home/qz/文档/Qz/workspace/ArtifactDataset/陈继金20170517 23-04-05'
    if os.path.exists(os.path.join(file_path, "raw_org.txt")) and os.path.exists(
            os.path.join(file_path, "Artifact_a.txt")):
        data_path = os.path.join(file_path, "raw_org.txt")  # 原始数据路径,"\\raw_org.txt"也可以
    elif os.path.exists(os.path.join(file_path, "new_org_1000hz.txt")) and os.path.exists(
            os.path.join(file_path, "Artifact_a.txt")):
        data_path = os.path.join(file_path, "new_org_1000hz.txt")  # 原始数据路径,"\\raw_org.txt"也可以
    else:
        print('数据或标签打开错误，不存在！')

    bcg_data = pd.read_csv(data_path, header=None, dtype=int).to_numpy().reshape(-1)  # 原始数据读取为numpy形式
    print('\033[1;31m Finished loading! \033[0m')

    bcg_data = sample_rate_change(bcg_data, change_rate=-10)

    start_point = 1500000
    bcg_data = bcg_data[start_point:start_point + 720000]  # 只取其中一部分，因为分割的速度慢

    bcg_data = Butterworth(bcg_data, type='bandpass', lowcut=2, highcut=15, order=2, Sample_org=100)
    # plt.plot(bcg_data)
    # plt.show()

    normal_signal = bcg_data
    artifact_signal = np.full(len(normal_signal), np.nan)  # 创建与newBCG一样长度的空数组

    print('\033[1;32m Spliting... \033[0m' % ())
    bcg_data = signal_split(bcg_data, split_step=1, split_len=5, sample_rate=100)
    print('\033[1;31m Finished spliting! \033[0m')

    bcg_data = torch.tensor(bcg_data, dtype=torch.float32)  # Tensor和tensor不一样

    bcg_data = bcg_data.reshape(bcg_data.shape[0], 1, bcg_data.shape[1])  # reshape成Cov1D的输入维度

    test_dataset = torch.utils.data.TensorDataset(bcg_data)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=nw)

    # create model
    model = vgg(model_name="vgg16", num_classes=2).to(device)
    # load model weights
    weights_path = "./VGGNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    predict_result = np.array([])  # 用来保存预测的结果，0/1

    model.eval()
    with torch.no_grad():
        # predict class
        test_bar = tqdm(train_loader)
        for test_data in test_bar:
            datas = test_data[0]  # test_data其实是List(iter)，List的每个元素里面是Tensor，所以不能直接整个赋值给datas
            outputs = model(datas.to(device))  # test_data的每个元素包含两个Tensor，第一个是数据，第二个是标签，所以价格[0]
            '''
            output = torch.max(input, dim)
            input是softmax函数输出的一个tensor,dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            '''
            # print(torch.max(outputs, dim=1)[1].cpu().numpy())
            # 因为是在GPU里面运算，所以torch.max返回最后有GPU信息，报错，加个.cpu()即可，另外转int可以直接加在最后面
            predict_result = np.append(predict_result, torch.max(outputs, dim=1)[1].cpu().numpy()).astype(int)

            # for i in range(len(datas)):
            #     print(datas.shape)
            #     plt.plot(datas[i, 0, :])
            #     plt.title('The predict result is : {}'.format(predict_result[i]))
            #     plt.show()

        print('Finished testing')

    # predict_result = predict_result.astype(int)
    print(predict_result)

    for i in range(len(predict_result)):
        if predict_result[i] == 1:
            artifact_signal[i * 100:(i + 5) * 100] = normal_signal[i * 100:(i + 5) * 100]
            # normal_signal[i * 100:(i + 5) * 100] = (np.nan,)

    normal_signal = normal_signal - artifact_signal  # 直接减，nan会出问题，无论作为减数还是被减数

    plt.plot(artifact_signal, color='red', label="大体动")
    plt.plot(normal_signal, color='green', label="正常数据")
    plt.legend(ncol=2)

    plt.show()

    print('Total program execution time = %2d min : %2ds' % (
        (time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))


if __name__ == '__main__':
    main()
