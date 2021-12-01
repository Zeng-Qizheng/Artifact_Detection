import torch.nn as nn
import torch


def get_parameter_number(net):
    '''
    numel()函数：返回数组中元素的个数
    :params = sum(p.numel() for p in list(net.parameters())) / 1e6 # numel()
    :print('#Params: %.1fM' % (params))
    net.parameters()：是Pytorch用法，用来返回net网络中的参数
    params：用来返回net网络中的参数的总数目
    '''
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=500,  # 图片每行的数据像素点
            hidden_size=128,  # rnn hidden unit
            num_layers=2,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, 2)  # 输出层

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()

        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out


# official pretrain weights
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
# }


# class VGG(nn.Module):
#     def __init__(self, features, num_classes=1000, init_weights=False):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             # nn.Linear(512*7*7, 4096),
#             # nn.Linear(512 * 15, 4096),
#             nn.Linear(16*15, 128),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 64),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(64, num_classes)
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.features(x)
#         print("The shape of x after features is :",x.shape)
#         # N x 512 x 7 x 7
#         x = torch.flatten(x, start_dim=1)
#         # N x 512*7*7
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 # nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

class VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 1, kernel_size=24, padding=1),
            nn.BatchNorm1d(1),
            # nn.ReLU(True),
            # nn.Softmax(10)


        )
        self.classifier = nn.Sequential(
            # nn.Linear(512*7*7, 4096),
            # nn.Linear(512 * 15, 4096),
            nn.Linear(16*15, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # print("The shape of x after features is :",x.shape)
        # x = nn.Softmax(dim=1)(x)
        # print(x[:10])
        # print(torch.sum(x,dim=1))
        # N x 512 x 7 x 7
        # x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def make_features(cfg: list):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(conv1d.out_channels), nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16': [16, 'M', 32, 'M', 64, 'M', 32, 'M', 16, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FCN_model(nn.Module):
    def __init__(self, NumClassesOut, N_time, N_Features, N_LSTM_Out=128, N_LSTM_layers=1
                 , Conv1_NF=128, Conv2_NF=256, Conv3_NF=128,lstmDropP = 0.8,FC_DropP = 0.3):
        super(FCN_model, self).__init__()

        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features, self.N_LSTM_Out, self.N_LSTM_layers)
        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.NumClassesOut)


    def init_hidden(self):
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        return h0, c0


    def forward(self, x):
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features

        h0, c0 = self.init_hidden()
        x1, (ht, ct) = self.lstm(x, (h0, c0))
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)
        return x_out

