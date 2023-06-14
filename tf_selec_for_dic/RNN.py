import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, inputsize):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=inputsize,
            hidden_size=inputsize,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            nonlinearity='tanh',
            batch_first=True,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # self.out = nn.Linear(32, 1)
        self.out1 = nn.Linear(inputsize, inputsize)
        self.out2 = nn.Sigmoid()

    def forward(self, x, h_state):
        r_out, h_state_next = self.rnn(x, h_state)
        r_out = self.out1(r_out)
        r_out = self.out2(r_out)
        return r_out, h_state_next


class RNN_class(nn.Module):
    def __init__(self, inputsize):
        super(RNN_class, self).__init__()

        self.rnn = nn.RNN(
            input_size=inputsize,
            hidden_size=128,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            nonlinearity='tanh',
            batch_first=True,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # self.out = nn.Linear(32, 1)
        self.out1 = nn.Linear(128, 13)
        self.out2 = nn.Softmax(dim=1)

    def forward(self, x, h_state):
        r_out, h_state_next = self.rnn(x, h_state)
        # print('one',r_out.shape,r_out)
        # print('oneone',h_state_next)
        r_out = r_out[:, -1, :]  # #选取最后一个时间的out作为评价
        # print('two', r_out.shape, r_out)
        r_out = self.out1(r_out)
        # print('three', r_out.shape, r_out)
        r_out = self.out2(r_out)
        # print('four', r_out.shape, r_out)
        return r_out, h_state_next

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3),(1,1), (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024, 13),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.cnn5(out)
        out = self.cnn6(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


class CNN_end_to_end_three(nn.Module):
    def __init__(self):
        super(CNN_end_to_end_three, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 2, 1 , 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 2, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # self.cnn4 = nn.Sequential(
        #     nn.Conv2d(256, 256, 2, 1, 0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.MaxPool2d((1, 1), (1, 1), 0),
        # )
        self.fc = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
        self.fc5 =nn.Sequential(
            nn.Linear(256*1*1,4096),
            nn.ReLU(),
            # nn.Dropout(p=0.3)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(4096,2048),
            nn.ReLU(),
            # nn.Dropout(p=0.3)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            # nn.Dropout(p=0.3)
        )

        self.fc8 = nn.Sequential(
            nn.Linear(1024, 181),
            nn.Softmax(dim=1),
        )


    def forward(self, x):
        out = self.cnn1(x)
        print('1',out.shape)
        out = self.fc(out)
        print('2', out.shape)
        return out

class Model_test(nn.Module):
    def __init__(self):
        super(Model_test, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,256, 3, 2, 0), # [64, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0), # [64, 64, 64]

            nn.Conv2d(256, 256, 2, 1, 0), # [128, 64, 64]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0), # [128, 32, 32]

            nn.Conv2d(256, 256, 2, 1, 0), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0), # [256, 16, 16]
        )
        self.fc = nn.Sequential(
            nn.Linear(256*1*1, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 181)
        )

    def forward(self, x):
        x = self.cnn(x)
        # print('x',x.shape)
        # print('x size', x)
        x = x.view(x.size()[0], -1) # torch.nn只支持mini-batches而不支持单个sample，第1个维度是mini-batch中图片（特征）的索引，即将每张图片都展开
        # print('x view size', x)
        return self.fc(x)


class RNN_DRR(nn.Module):
    def __init__(self, inputsize):
        super(RNN_DRR, self).__init__()
        self.rnn = nn.GRU(
            input_size=inputsize,
            hidden_size=129*2,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            nonlinearity='tanh',
            batch_first=True,
        )
        self.tower1 = nn.Sequential(
            nn.Linear(129*2, 129),
            nn.ReLU(),
            nn.Linear(129, 129),
            nn.Sigmoid(),
        )
        self.tower2 = nn.Sequential(
            nn.Linear(129*2, 129),
            nn.ReLU(),
            nn.Linear(129, 129),
        )


    def forward(self, x, h_state):
        r_out0, h_state = self.rnn(x, h_state)
        r_out1 = self.tower1(r_out0)
        r_out2 = self.tower2(r_out0)
        return r_out1, r_out2, h_state


class RNN_DRR_IRM(nn.Module):
    def __init__(self, inputsize):
        super(RNN_DRR_IRM, self).__init__()
        self.rnn = nn.RNN(
            input_size=inputsize,
            hidden_size=43*8*3,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            nonlinearity='tanh',
            batch_first=True,
        )
        self.tower1 = nn.Sequential(
            nn.Linear(43*8*3, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.Sigmoid(),
        )
        self.tower2 = nn.Sequential(
            nn.Linear(43*8*3, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.Sigmoid(),
        )
        self.tower3 = nn.Sequential(
            nn.Linear(43*8*3, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.Sigmoid(),
        )

    def forward(self, x, h_state):
        r_out0, h_state = self.rnn(x, h_state)
        r_out1 = self.tower1(r_out0)
        r_out2 = self.tower2(r_out0)
        r_out3 = self.tower3(r_out0)

        return r_out1, r_out2, r_out3, h_state

class RNN_DRR_SPP(nn.Module):
    def __init__(self, inputsize):
        super(RNN_DRR_SPP, self).__init__()
        self.rnn = nn.GRU(
            input_size=inputsize,
            hidden_size=43*8*2,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            # nonlinearity='tanh',
            batch_first=True,
        )
        self.tower1 = nn.GRU(
            input_size = 43*8*2,
            hidden_size = 43*8,
            num_layers = 2,
            batch_first=True,
        )
        
        self.tower1_2=nn.Sequential(
            nn.Linear(43*8,43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.Sigmoid(),
        )
        self.tower2 = nn.GRU(
            input_size=43*8*2,
            hidden_size=43*8,
            num_layers=2,
            batch_first=True,    
        )
        self.tower2_2=nn.Sequential(
            nn.Linear(43*8,43*8),
            nn.ReLU(),
            nn.Linear(43*8, 43*8),
            nn.Sigmoid(),
        )
        

    def forward(self, x, h_state,h_state1,h_state2):
        r_out0, h_state  = self.rnn(x, h_state)
        r_out1, h_state1 = self.tower1(r_out0,h_state1)
        r_out1_2   =self.tower1_2(r_out1)
        r_out2, h_state2 = self.tower2(r_out0,h_state2)
        r_out2_2   =self.tower2_2(r_out2)
        return r_out1_2, r_out2_2, h_state, h_state1, h_state2


class RNN_TWO_GET_DRR(nn.Module):
    def __init__(self, inputsize):
        super(RNN_TWO_GET_DRR, self).__init__()
        self.rnn = nn.GRU(
            input_size=inputsize,
            hidden_size=43 * 8 ,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            # nonlinearity='tanh',
            batch_first=True,
        )
        self.tower1 = nn.Sequential(
            nn.Linear(43 * 8, 43 * 8),
            nn.Sigmoid(),
        )


    def forward(self, x, h_state):
        r_out0, h_state = self.rnn(x, h_state)
        r_out1 = self.tower1(r_out0)

        return r_out1, h_state

class RNN_SPP_contain_rever(nn.Module):
    def __init__(self, inputsize):
        super(RNN_SPP_contain_rever, self).__init__()
        self.rnn = nn.GRU(
            input_size=inputsize,
            hidden_size=43 * 8 ,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            # nonlinearity='tanh',
            batch_first=True,
        )
        self.tower1 = nn.Sequential(
            nn.Linear(43 * 8, 43 * 8),
            nn.ReLU(),
            nn.Linear(43 * 8, 43 * 8),
            nn.Sigmoid(),
        )


    def forward(self, x, h_state):
        r_out0, h_state = self.rnn(x, h_state)
        r_out1 = self.tower1(r_out0)
        return r_out1, h_state
