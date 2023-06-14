#需要试一下时间作为维度吗？ 其他几个参数的设置有讲究吗？ 特别是h_hiden size   h0的输出值

import gc
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import RNN
# import file_load2
import MatrixDataset2
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import time
# import SPP_plot
from torch.autograd import Variable

from DRR_IRM_fenpiduqu import readfile_fenpi_input_43yipi_duoyi, readfile_fenpi_output_43yipi_duoyi

# torch.manual_seed(1)    # reproducible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(device)

# matfn1='D:\\database\\train'
# matfn2='D:\\database\\target_train'
# matfn3='D:\\database\\test'
# matfn4='D:\\database\\target_test'
num_epoch=151

fs = 16000
jiequshijian = 2
batch_size = 128

t_saperate = 16

LR =1*1e-4  # learning rate
dimen5_train = int(249/t_saperate)
#任意取一个文件，看看大小   #选target里面的大小

####Load hdf5 dataset
for frequency in range(0,3,1):
    print('frequency',frequency)
    #####读取train
    list_title_train = os.listdir('/home/create.aau.dk/su41my/SPP_contain_rever_SER/train')
    url_train = '/home/create.aau.dk/su41my/SPP_contain_rever_SER/train/'
    type1 = 'x_train'
    train_x,train_inputdimen = readfile_fenpi_input_43yipi_duoyi(list_title_train,url_train,frequency,type1)


    print('total_final_input',train_x.shape)

    type2 = 'target_SPP'
    train_SPP_target = readfile_fenpi_output_43yipi_duoyi(list_title_train, url_train, frequency, type2)
    # type3 = 'target_DRR'
    # train_DRR_target = readfile_fenpi_output_43yipi_duoyi(list_title_train, url_train, frequency, type3)


    train_set = MatrixDataset2.matrixDataset_TWO_GET_DRR(train_x, train_SPP_target)
    del train_x, train_SPP_target
    gc.collect()
    #######读取test

    list_title_test = os.listdir('/home/create.aau.dk/su41my/SPP_contain_rever_SER/test')
    url_test = '/home/create.aau.dk/su41my/SPP_contain_rever_SER/test/'
    type4 = 'x_test'
    test_x,test_inputdimen = readfile_fenpi_input_43yipi_duoyi(list_title_test, url_test, frequency, type4)

    # test_x = np.concatenate((test_x, test_x_onlyhun), axis=2)
    # del test_x_onlyhun
    # gc.collect()
    print('total_final_input', test_x.shape)
    # test_inputdimen = test_inputdimen + test_inputdimen_2
    type5 = 'target_SPP'
    test_SPP_target = readfile_fenpi_output_43yipi_duoyi(list_title_test, url_test, frequency, type5)

    # type6 = 'target_DRR'
    # test_DRR_target = readfile_fenpi_output_43yipi_duoyi(list_title_test, url_test, frequency, type6)


    ##############3
    test_set = MatrixDataset2.matrixDataset_TWO_GET_DRR(test_x, test_SPP_target)
    del test_x,test_SPP_target
    gc.collect()


    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle=False, drop_last=True)

    train_length = train_set.__len__()
    test_length = test_set.__len__()
    del train_set
    del test_set
    gc.collect()

    # model = RNN.RNN(INPUT_SIZE).cuda()
    model = RNN.RNN_SPP_contain_rever(train_inputdimen).cuda()
    # model = torch.load("modelpara_veri_16batch_200epoch_11_21_noise92_8cm_016_lowSNR_yang.pth")

    print(model)


    # optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=0.0001)   # optimize all cnn parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(num_epoch):
        hidden_state = None  # for initial hidden state
        # hidden_state1= None
        # hidden_state2= None
        # hidden_state = torch.zeros([])
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0

        train_loss1 = 0.0
        train_loss2 = 0.0
        
        test_loss1 = 0.0
        test_loss2 = 0.0
        
    # 网络训练
        model.train()
        for i, data in enumerate(train_loader):
            # prediction = model(data[0].cuda())
            mean_matrix = torch.mean(data[0].to(device))
            std_matrix = torch.std(data[0].to(device))
            matrix = (data[0].to(device) - mean_matrix) / (std_matrix)

            optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零
            prediction_SPP, hidden_state = model(matrix, hidden_state)
            # print('shape对比',prediction.shape,data[1].cpu().shape)
            # print('data0 shape',data[0].cpu().shape)   #32(N)*16(T)*1032(F)
            # hidden_state = hidden_state.data
            hidden_state = Variable(hidden_state.data).to(device)
            # hidden_state1 = Variable(hidden_state1.data).to(device)
            # hidden_state2 = Variable(hidden_state2.data).to(device)

            batch_loss = batch_size * loss_func(prediction_SPP, data[1].to(device))
            # batch_loss_2 = batch_size * loss_func(prediction_DRR, data[2].to(device))
            # batch_loss_sum =  0.9 * batch_loss_1 + 0.1 * batch_loss_2 / (batch_loss_2 / batch_loss_1).detach()
            # batch_loss = batch_loss_sum
            
            # batch_loss = 0.5 * batch_loss_1 + 0.5 * batch_loss_2
            # print('loss',batch_loss)

            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)   ####加上梯度裁决避免梯度爆炸
            optimizer.step()         # 以 optimizer 用 gradient 更新参数
            # scheduler.step()

            train_loss += batch_loss.item()               #.item()方法 是得到一个元素张量里面的元素值
            # train_loss1 += batch_loss_1.item()
            # train_loss2 += batch_loss_2.item()
    # 网络测试
        hidden_state_test = None  # for initial hidden state
        # hidden_state_test1= None
        # hidden_state_test2= None

        model.eval()
        with torch.no_grad():
            for i_test, data_test in enumerate(test_loader):
                mean_matrix_test = torch.mean(data_test[0].to(device))
                std_matrix_test = torch.std(data_test[0].to(device))
                matrix_test = (data_test[0].to(device) - mean_matrix_test) / (std_matrix_test)

                prediction_test_SPP, hidden_state_test = model(matrix_test, hidden_state_test)
                # print('data size', data_test[0].cpu().shape)
                batch_loss_test = batch_size * loss_func(prediction_test_SPP, data_test[1].to(device))
                # batch_loss_2_test = batch_size * loss_func(prediction_test_DRR, data_test[2].to(device))
                # batch_loss_sum_test = 0.9 * batch_loss_1_test + 0.1 * batch_loss_2_test / (batch_loss_2_test / batch_loss_1_test).detach()
                # batch_loss_test = batch_loss_sum_test

                test_loss  += batch_loss_test.item()
                # test_loss1 += batch_loss_1_test.item()
                # test_loss2 += batch_loss_2_test.item()
            if epoch % 5 == 0:
                model_name = os.path.join('SPP_containrever_noweightdecay_-5_%dfrequency_%depoch'%(frequency,epoch))
                torch.save(model, model_name)
                #     prediction_test = prediction_test.permute(1,2,0)
                    # print('prediction size',prediction_test.shape)
                    # if i_test == 0:
                    #     prediction_hecheng = prediction_test
                    # else:
                    #     prediction_hecheng = torch.cat((prediction_hecheng, prediction_test), axis=2)   ###这里的合成是在合成batch，第三维度现在是总数量，不是时域或者频域

            print('[%03d/%03d] %2.2f sec(s) Loss: %3.6f | loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                   train_loss / train_length, test_loss / test_length))
            # print(' Loss_spp: %3.6f Loss_drr: %3.6f | loss_spp: %3.6f loss_drr: %3.6f'  % \
            #        (train_loss1 / train_length, train_loss2 / train_length, \
            #         test_loss1 / test_length, test_loss2 / test_length))
            #
    del train_loader,test_loader
    gc.collect()
    #训练完毕后，取出生成的数

    #下面是100个数据做验证
    #     if epoch ==num_epoch-1:
    #         hidden_state_verify = None
    #         with torch.no_grad():
    #             for i_verify, data_verify in enumerate(verify_loader):
    #                 # print('data verify0 shape', data_verify[0].cpu().shape)  #1 1032?
    #                 prediction_verify, hidden_state_verify = model(data_verify[0].cpu(), hidden_state_verify)
    #                 # print('data size', data_test[0].cpu().shape)
    #
    #                 prediction_verify = prediction_verify.permute(2,1,0)
    #                 # print('prediction_verify shape', prediction_verify.shape)
    #                 if i_verify == 0:
    #                     prediction_verify_total = prediction_verify
    #                 else:
    #                     prediction_verify_total = torch.cat((prediction_verify_total, prediction_verify), axis=2)   ###这里的合成是在合成batch，第三维度现在是总数量，不是时域或者频域
    # #
    # spp_matrix = prediction_verify_total.cpu().numpy()
    # print('spp',type(spp_matrix))
    # print('spp',spp_matrix.shape)
    #
    # maxmax = np.max(spp_matrix)
    # print('sppmax',maxmax)
    # # spp_backtranspose = np.transpose(spp_matrix, (1, 0, 2))
    # # print('spp_backtranspose.shape',spp_backtranspose)
    # # print('dimen5_target',dimen5_target)
    # # S_matrix_hecheng = SPP_plot.pinjie(spp_matrix,dimen5_target)
    # # print('S_matrix_hecheng',S_matrix_hecheng)
    # # print('S_matrix_hecheng.shape',S_matrix_hecheng.shape)
    # # SPP_plot.spp_plot(spp_matrix[:,:,0], fs, jiequshijian)
    # SPP_plot.spp_plot(spp_matrix[0:128,:,0], fs, jiequshijian)
    # SPP_plot.spp_plot(spp_matrix[0:128,:,3], fs, jiequshijian)
    # # SPP_plot.spp_plot(spp_matrix[903:1032,:,0], fs, jiequshijian)
    #
    #
    # verify_target = np.transpose(verify_target, (2, 1, 0))
    # SPP_plot.spp_plot(verify_target[0:128,:,0], fs, jiequshijian)
    # SPP_plot.spp_plot(verify_target[0:128,:,3], fs, jiequshijian)
    #
    #
    # print('tu.shape',spp_matrix.shape)
    # print('tu',spp_matrix[:,:,0])
    #
    # h5f = h5py.File('D:\\database\\result_fit2_-35db_016_noise92_100.h5', 'w')
    # h5f.create_dataset('result_SPP_0db', data= prediction_verify_total)
    # h5f.close()


    ###关于z的维度， 在66行，和read file 的z = x
    #  'modelpara_veri_32batch_200epoch_10_24.pth' 训练了200轮的0.2米的阵列 噪声没固定角度
