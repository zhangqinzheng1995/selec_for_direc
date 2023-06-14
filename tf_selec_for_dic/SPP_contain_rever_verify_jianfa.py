
import scipy as scipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import RNN
import file_load2
import MatrixDataset2
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import time
import SPP_plot
from torch.autograd import Variable
import noise_spp
from DRR_IRM_fenpiduqu import readfile_fenpi_input_43yipi_verify_duoyi, readfile_fenpi_output_43yipi_verify_duoyi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
fs=16000
jiequshijian = 0.3

    # print(device)
# for xinzao in range(-20,-14,5):
def xuandade(a):
    [changdu, kuandu]=a.shape
    cx = np.zeros([changdu,kuandu])
    for i in range(changdu):
        for j in range(kuandu):
            if a[i,j]>=0:
                cx[i,j]=a[i,j]
            else:
                cx[i,j]=0
    return cx


for xinzao in range(-25, -10+1, 5):
    for hunxiang in range(20,71,20):
    # for hunxiang in range(20, 71, 10):
        for jiaodu in range(2,13,1):
        # for jiaodu in range(2, 13, 1):
            SPP_final= 10 * torch.ones(1032, 36, 250)
            # DRR_final=10*torch.ones(1032,36,250)
            # PRODUCT_final=10*torch.ones(1032,61,500)
            # print('max',torch.max(IRM_final))
            # print('max',IRM_final.shape)
            for frequency in range(3):
                h5f_file = h5py.File('E:\\SPP_contain_rever\\verifydata_%ddb_0%d_noise92_100_05s_ang%d.h5' % (xinzao, hunxiang, jiaodu), 'r')

                type1 = 'x_verify'
                verify_x, verify_inputdimen = readfile_fenpi_input_43yipi_verify_duoyi(h5f_file,frequency,type1)
                # print('x_verify.shape', verify_x.shape)
                type2 = 'target_SPP'
                verify_SPP_target = readfile_fenpi_output_43yipi_verify_duoyi(h5f_file,frequency,type2)
                # print('verify_IRM_target',verify_IRM_target.shape)
                # type3 = 'target_DRR'
                # verify_DRR_target = readfile_fenpi_output_43yipi_verify_duoyi(h5f_file,frequency,type3)
                # maxmax_drdrdr=np.max(verify_DRR_target)
                # print('maxmax_drdrdr',maxmax_drdrdr)
                # print('verify_DRR_target',verify_DRR_target.shape)
                # verify_product_target = verify_IRM_target * verify_DRR_target
                INPUT_SIZE = verify_inputdimen

                verify_set= MatrixDataset2.matrixDataset_TWO_GET_DRR(verify_x, verify_SPP_target)
                verify_loader = DataLoader(verify_set, batch_size = 1, shuffle=False, drop_last=False)
                # model = RNN.RNN_DRR_SPP(INPUT_SIZE).cuda()
                model_name = os.path.join('SPP_containrever_noweightdecay_-5_%dfrequency_%depoch' % (frequency, 85))
                model = torch.load(model_name)

                model.eval()
                hidden_state_verify = None
                # hidden_state_verify1 = None
                # hidden_state_verify2 = None
                with torch.no_grad():
                    for i_verify, data_verify in enumerate(verify_loader):
                        mean_matrix_verify = torch.mean(data_verify[0].to(device))
                        std_matrix_verify = torch.std(data_verify[0].to(device))
                        matrix_verify = (data_verify[0].to(device) - mean_matrix_verify) / (std_matrix_verify)

                        prediction_verify_SPP, hidden_state_verify = model(matrix_verify, hidden_state_verify)
                        # prediction_SPP, prediction_DRR, hidden_state, hidden_state1, hidden_state2 = model(matrix_verify, hidden_state, hidden_state1, hidden_state2)
                        # print('这里可能需要扩张成三维,得看看大小才能继续编')
                        # print('prediction_verify_IRM',prediction_verify_IRM.shape)
                        # print('prediction_verify_DRR',prediction_verify_DRR.shape)
                        # print('prediction_verify_production',prediction_verify_PRODUCT.shape)
                        prediction_verify_SPP = prediction_verify_SPP.permute(2, 1, 0)
                        # prediction_verify_DRR = prediction_verify_DRR.permute(2, 1, 0)
                        # prediction_verify_DRR = prediction_verify_DRR.permute(2, 1, 0)
                        # prediction_verify_PRODUCT=prediction_verify_PRODUCT.permute(2,1,0)
                        # print('prediction_verify_IRM', prediction_verify_IRM.shape)
                        # print('prediction_verify_DRR', prediction_verify_DRR.shape)
                        # print('prediction_verify_production', prediction_verify_PRODUCT.shape)

                        if i_verify == 0:
                            prediction_verify_total_SPP = prediction_verify_SPP
                            # prediction_verify_total_DRR = prediction_verify_DRR
                            # prediction_verify_total_DRR = prediction_verify_DRR
                            # prediction_verify_total_PRODUCT = prediction_verify_PRODUCT
                        else:
                            prediction_verify_total_SPP = torch.cat((prediction_verify_total_SPP, prediction_verify_SPP), axis=2)   ###这里的合成是在合成batch，第三维度现在是总数量，不是时域或者频域
                            # prediction_verify_total_DRR = torch.cat((prediction_verify_total_DRR, prediction_verify_DRR), axis=2)
                            # prediction_verify_total_DRR = torch.cat((prediction_verify_total_DRR, prediction_verify_DRR), axis=2)  ###这里的合成是在合成batch，第三维度现在是总数量，不是时域或者频域
                            # prediction_verify_total_PRODUCT=torch.cat((prediction_verify_total_PRODUCT, prediction_verify_PRODUCT), axis=2)
                        ####上面是拼接个数，把所有个数都拼了d但是还没拼维度
                # print('这里再看看IRM与DRR的大小，以方便拼接维度')
                # print('prediction_verify_total_IRM',prediction_verify_total_IRM.shape)
                # print('prediction_verify_total_DRR', prediction_verify_total_DRR.shape)
                # print('prediction_verify_total_PRODUCT', prediction_verify_total_PRODUCT.shape)
                if frequency ==0:
                    for kk in range(8):
                        SPP_final[kk * 129: kk * 129 + 43, :, :] = prediction_verify_total_SPP[kk * 43:(kk + 1) * 43, :, :]
                        # DRR_final[kk * 129: kk * 129 + 43, :, :] = prediction_verify_total_DRR[kk * 43:(kk + 1) * 43, :, :]
                        # DRR_final[kk*129 : kk*129+43, :, :] = prediction_verify_total_DRR[kk*43:(kk+1)*43, :, :]
                        # PRODUCT_final[kk*129 : kk*129+43, :, :] = prediction_verify_total_PRODUCT[kk*43:(kk+1)*43, :, :]
                elif frequency==1:
                    for kk in range(8):
                        SPP_final[kk * 129 + 43 * 1:kk * 129 + 43 * 2, :, :] = prediction_verify_total_SPP[kk * 43:(kk + 1) * 43, :, :]
                        # DRR_final[kk * 129 + 43 * 1:kk * 129 + 43 * 2, :, :] = prediction_verify_total_DRR[kk * 43:(kk + 1) * 43, :, :]
                        # DRR_final[kk*129+43*1:kk*129+43*2, :, :] = prediction_verify_total_DRR[kk*43:(kk+1)*43, :, :]
                        # PRODUCT_final[kk*129+43*1:kk*129+43*2, :, :] = prediction_verify_total_PRODUCT[kk*43:(kk+1)*43, :, :]
                else:
                    for kk in range(8):
                        SPP_final[kk * 129 + 43 * 2:kk * 129 + 43 * 3, :, :] = prediction_verify_total_SPP[kk * 43:(kk + 1) * 43, :, :]
                        # DRR_final[kk * 129 + 43 * 2:kk * 129 + 43 * 3, :, :] = prediction_verify_total_DRR[kk * 43:(kk + 1) * 43, :, :]

                        # DRR_final[kk*129+43*2:kk*129+43*3, :, :] = prediction_verify_total_DRR[kk*43:(kk+1)*43, :, :]
                        # PRODUCT_final[kk*129+43*2:kk*129+43*3, :, :] = prediction_verify_total_PRODUCT[kk*43:(kk+1)*43, :, :]

            spp_matrix = SPP_final.cpu().numpy()
            # drr_matrix = DRR_final.cpu().numpy()
            # drr_matrix = DRR_final.cpu().numpy()
            # product_matrix = PRODUCT_final.cpu().numpy()

            # print('spp',type(spp_matrix))
            print('spp', spp_matrix.shape)
            # print('drr', drr_matrix.shape)
            # print('drr',drr_matrix.shape)
            # print('product',product_matrix.shape)

            maxmax_spp = np.max(spp_matrix)
            print('sppmax',maxmax_spp)
            # maxmax_drr = np.max(drr_matrix)
            # print('drrmax',maxmax_drr)
            # maxmax_product = np.max(product_matrix)
            # print('product',maxmax_product)
                # spp_backtranspose = np.transpose(spp_matrix, (1, 0, 2))
                # print('spp_backtranspose.shape',spp_backtranspose)
                # print('dimen5_target',dimen5_target)
                # S_matrix_hecheng = SPP_plot.pinjie(spp_matrix,dimen5_target)
                # print('S_matrix_hecheng',S_matrix_hecheng)
                # print('S_matrix_hecheng.shape',S_matrix_hecheng.shape)
                # SPP_plot.spp_plot(spp_matrix[:,:,0], fs, jiequshijian)
                # SPP_plot.spp_plot(spp_matrix[0:128,:,0], fs, jiequshijian)
                # SPP_plot.spp_plot(spp_matrix[0:128,:,3], fs, jiequshijian)
                # SPP_plot.spp_plot(spp_matrix[903:1032,:,0], fs, jiequshijian)


                # verify_IRM_target = np.transpose(verify_IRM_target, (2, 1, 0))
                # verify_DRR_target = np.transpose(verify_DRR_target, (2, 1, 0))
                # SPP_plot.spp_plot(verify_target[0:128,:,0], fs, jiequshijian)
                # SPP_plot.spp_plot(verify_target[0:128,:,3], fs, jiequshijian)

                ####是否看spp和drr
                # print('spptu.shape',spp_matrix.shape)
                # print('spptu',spp_matrix[:,:,0])
                # print('drrtu.shape',drr_matrix.shape)
                # print('drrtu',drr_matrix[:,:,0])

            # h5f = h5py.File('D:\\matlab2\\toolbox\\RIR-Generator-master\\DRR5_SPP_twogetone\\verify\\result_DRR_SPP_focusSPP_%ddb_0%d_ang%d.h5' %(xinzao,hunxiang,jiaodu), 'w')
            # h5f.create_dataset('result_SPP', data= IRM_final.cpu())
            # h5f.close()
            path = r'E:\\SPP_contain_rever\\'

            y_name = os.path.join(path, 'y_stft_%ddb_0%d_noise92_100_05s_ang%d' % (xinzao, hunxiang, jiaodu))
            y_stft_total = scipy.io.loadmat(y_name)['y_stft_total']
            y_name2 = os.path.join(path, 'y_%ddb_0%d_noise92_100_05s_ang%d' % (xinzao, hunxiang, jiaodu))
            y_total2 = scipy.io.loadmat(y_name2)['total_withnoise_signal']

            [pin, shi, tongdao, yshuliang] = y_stft_total.shape
            y_stft_final = np.zeros([pin, shi, tongdao, yshuliang], complex)

            for meiyige in range(yshuliang):
                for cc in range(tongdao):
                    y_stft = y_stft_total[:, :, cc, meiyige]  # %%看看y是不是提出来了
                    y_psd = np.abs(y_stft) ** 2

                    y_signal = y_total2[:, cc, meiyige]
                    SPP = spp_matrix[cc * 129:(cc + 1) * 129, :, meiyige]  ## %%看大小对不对
                    # print('SPP',SPP.shape)

                    noise_psd, snr = noise_spp.noisepowproposed_spp(y_signal, fs, SPP)
                    clean_psd_first = y_psd - noise_psd
                    clean_psd = xuandade(clean_psd_first)
                    ratio = clean_psd / y_psd  ## 看看是不是点除
                    # print('clean_psd_first',clean_psd_first)
                    y_new_stft = y_stft * ratio  ## 看看是不是对应位相乘
                    y_stft_final[:, :, cc, meiyige] = y_new_stft
            print('y_stft_final', y_stft_final)
            matlabname = 'E:\\SPP_contain_rever\\after_jianfa\\result_SPP_contain_rever_jianfa_stft_focusSPP_%ddb_0%d_ang%d.mat' % (xinzao, hunxiang, jiaodu)
            scipy.io.savemat(matlabname, {'y_stft_final': y_stft_final})

# A = np.arange(8).reshape((2,2,2))


