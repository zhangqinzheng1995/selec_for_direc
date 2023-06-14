import h5py
import numpy as np
def readfile_fenpi_input_43yipi_verify_duoyi(h5f_file, frequency, type_name ):
    if frequency == 0:
        data_x0 = np.array(h5f_file[type_name])
        # print('data000',data_x0.shape)
        data_x1 = data_x0[:, :, 0+129*0:43+1+129*0]
        data_x2 = data_x0[:, :, 0+129*1:43+1+129*1]
        data_x3 = data_x0[:, :, 0+129*2:43+1+129*2]
        data_x4 = data_x0[:, :, 0+129*3:43+1+129*3]
        data_x5 = data_x0[:, :, 0+129*4:43+1+129*4]
        data_x6 = data_x0[:, :, 0+129*5:43+1+129*5]
        data_x7 = data_x0[:, :, 0+129*6:43+1+129*6]
        data_x8 = data_x0[:, :, 0+129*7:43+1+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 53 * 8
    elif  frequency == 2:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129*1-44 : 129*1]
        data_x2 = data_x0[:, :, 129*2-44 : 129*2]
        data_x3 = data_x0[:, :, 129*3-44 : 129*3]
        data_x4 = data_x0[:, :, 129*4-44 : 129*4]
        data_x5 = data_x0[:, :, 129*5-44 : 129*5]
        data_x6 = data_x0[:, :, 129*6-44 : 129*6]
        data_x7 = data_x0[:, :, 129*7-44 : 129*7]
        data_x8 = data_x0[:, :, 129*8-44 : 129*8]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 53 * 8
    else:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129*0+43-1 : 129*0+43*2+1]
        data_x2 = data_x0[:, :, 129*1+43-1 : 129*1+43*2+1]
        data_x3 = data_x0[:, :, 129*2+43-1 : 129*2+43*2+1]
        data_x4 = data_x0[:, :, 129*3+43-1 : 129*3+43*2+1]
        data_x5 = data_x0[:, :, 129*4+43-1 : 129*4+43*2+1]
        data_x6 = data_x0[:, :, 129*5+43-1 : 129*5+43*2+1]
        data_x7 = data_x0[:, :, 129*6+43-1 : 129*6+43*2+1]
        data_x8 = data_x0[:, :, 129*7+43-1 : 129*7+43*2+1]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 63*8
    return data_final,inputsize

def readfile_fenpi_output_43yipi_verify_duoyi(h5f_file, frequency, type_name ):
    if frequency == 0:
        data_x0 = np.array(h5f_file[type_name])
        # print('data000',data_x0.shape)
        data_x1 = data_x0[:, :, 0+129*0: 43+129*0]
        data_x2 = data_x0[:, :, 0+129*1: 43+129*1]
        data_x3 = data_x0[:, :, 0+129*2: 43+129*2]
        data_x4 = data_x0[:, :, 0+129*3: 43+129*3]
        data_x5 = data_x0[:, :, 0+129*4: 43+129*4]
        data_x6 = data_x0[:, :, 0+129*5: 43+129*5]
        data_x7 = data_x0[:, :, 0+129*6: 43+129*6]
        data_x8 = data_x0[:, :, 0+129*7: 43+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
    elif frequency == 2:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129 * 1 - 43: 129 * 1]
        data_x2 = data_x0[:, :, 129 * 2 - 43: 129 * 2]
        data_x3 = data_x0[:, :, 129 * 3 - 43: 129 * 3]
        data_x4 = data_x0[:, :, 129 * 4 - 43: 129 * 4]
        data_x5 = data_x0[:, :, 129 * 5 - 43: 129 * 5]
        data_x6 = data_x0[:, :, 129 * 6 - 43: 129 * 6]
        data_x7 = data_x0[:, :, 129 * 7 - 43: 129 * 7]
        data_x8 = data_x0[:, :, 129 * 8 - 43: 129 * 8]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
        inputsize = 53 * 8
    else:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 43+129*0 : 129*1-43]
        data_x2 = data_x0[:, :, 43+129*1 : 129*2-43]
        data_x3 = data_x0[:, :, 43+129*2 : 129*3-43]
        data_x4 = data_x0[:, :, 43+129*3 : 129*4-43]
        data_x5 = data_x0[:, :, 43+129*4 : 129*5-43]
        data_x6 = data_x0[:, :, 43+129*5 : 129*6-43]
        data_x7 = data_x0[:, :, 43+129*6 : 129*7-43]
        data_x8 = data_x0[:, :, 43+129*7 : 129*8-43]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
    return data_final
###
def readfile_fenpi_input_43yipi_duoyi(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0:43+1+129*0]
            data_x2 = data_x0[:, :, 0+129*1:43+1+129*1]
            data_x3 = data_x0[:, :, 0+129*2:43+1+129*2]
            data_x4 = data_x0[:, :, 0+129*3:43+1+129*3]
            data_x5 = data_x0[:, :, 0+129*4:43+1+129*4]
            data_x6 = data_x0[:, :, 0+129*5:43+1+129*5]
            data_x7 = data_x0[:, :, 0+129*6:43+1+129*6]
            data_x8 = data_x0[:, :, 0+129*7:43+1+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 44 * 8
        elif  frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*1-44 : 129*1]
            data_x2 = data_x0[:, :, 129*2-44 : 129*2]
            data_x3 = data_x0[:, :, 129*3-44 : 129*3]
            data_x4 = data_x0[:, :, 129*4-44 : 129*4]
            data_x5 = data_x0[:, :, 129*5-44 : 129*5]
            data_x6 = data_x0[:, :, 129*6-44 : 129*6]
            data_x7 = data_x0[:, :, 129*7-44 : 129*7]
            data_x8 = data_x0[:, :, 129*8-44 : 129*8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 44 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*0+43-1 : 129*0+43*2+1]
            data_x2 = data_x0[:, :, 129*1+43-1 : 129*1+43*2+1]
            data_x3 = data_x0[:, :, 129*2+43-1 : 129*2+43*2+1]
            data_x4 = data_x0[:, :, 129*3+43-1 : 129*3+43*2+1]
            data_x5 = data_x0[:, :, 129*4+43-1 : 129*4+43*2+1]
            data_x6 = data_x0[:, :, 129*5+43-1 : 129*5+43*2+1]
            data_x7 = data_x0[:, :, 129*6+43-1 : 129*6+43*2+1]
            data_x8 = data_x0[:, :, 129*7+43-1 : 129*7+43*2+1]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 45*8
    return data_total,inputsize

def readfile_fenpi_output_43yipi_duoyi(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0: 43+129*0]
            data_x2 = data_x0[:, :, 0+129*1: 43+129*1]
            data_x3 = data_x0[:, :, 0+129*2: 43+129*2]
            data_x4 = data_x0[:, :, 0+129*3: 43+129*3]
            data_x5 = data_x0[:, :, 0+129*4: 43+129*4]
            data_x6 = data_x0[:, :, 0+129*5: 43+129*5]
            data_x7 = data_x0[:, :, 0+129*6: 43+129*6]
            data_x8 = data_x0[:, :, 0+129*7: 43+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
        elif frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129 * 1 - 43: 129 * 1]
            data_x2 = data_x0[:, :, 129 * 2 - 43: 129 * 2]
            data_x3 = data_x0[:, :, 129 * 3 - 43: 129 * 3]
            data_x4 = data_x0[:, :, 129 * 4 - 43: 129 * 4]
            data_x5 = data_x0[:, :, 129 * 5 - 43: 129 * 5]
            data_x6 = data_x0[:, :, 129 * 6 - 43: 129 * 6]
            data_x7 = data_x0[:, :, 129 * 7 - 43: 129 * 7]
            data_x8 = data_x0[:, :, 129 * 8 - 43: 129 * 8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
            inputsize = 53 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 43+129*0 : 129*1-43]
            data_x2 = data_x0[:, :, 43+129*1 : 129*2-43]
            data_x3 = data_x0[:, :, 43+129*2 : 129*3-43]
            data_x4 = data_x0[:, :, 43+129*3 : 129*4-43]
            data_x5 = data_x0[:, :, 43+129*4 : 129*5-43]
            data_x6 = data_x0[:, :, 43+129*5 : 129*6-43]
            data_x7 = data_x0[:, :, 43+129*6 : 129*7-43]
            data_x8 = data_x0[:, :, 43+129*7 : 129*8-43]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
    return data_total


###
def readfile_fenpi_input_43yipi_verify(h5f_file, frequency, type_name ):
    if frequency == 0:
        data_x0 = np.array(h5f_file[type_name])
        # print('data000',data_x0.shape)
        data_x1 = data_x0[:, :, 0+129*0:43+10+129*0]
        data_x2 = data_x0[:, :, 0+129*1:43+10+129*1]
        data_x3 = data_x0[:, :, 0+129*2:43+10+129*2]
        data_x4 = data_x0[:, :, 0+129*3:43+10+129*3]
        data_x5 = data_x0[:, :, 0+129*4:43+10+129*4]
        data_x6 = data_x0[:, :, 0+129*5:43+10+129*5]
        data_x7 = data_x0[:, :, 0+129*6:43+10+129*6]
        data_x8 = data_x0[:, :, 0+129*7:43+10+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 53 * 8
    elif  frequency == 2:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129*1-53 : 129*1]
        data_x2 = data_x0[:, :, 129*2-53 : 129*2]
        data_x3 = data_x0[:, :, 129*3-53 : 129*3]
        data_x4 = data_x0[:, :, 129*4-53 : 129*4]
        data_x5 = data_x0[:, :, 129*5-53 : 129*5]
        data_x6 = data_x0[:, :, 129*6-53 : 129*6]
        data_x7 = data_x0[:, :, 129*7-53 : 129*7]
        data_x8 = data_x0[:, :, 129*8-53 : 129*8]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 53 * 8
    else:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129*0+43-10 : 129*0+43*2+10]
        data_x2 = data_x0[:, :, 129*1+43-10 : 129*1+43*2+10]
        data_x3 = data_x0[:, :, 129*2+43-10 : 129*2+43*2+10]
        data_x4 = data_x0[:, :, 129*3+43-10 : 129*3+43*2+10]
        data_x5 = data_x0[:, :, 129*4+43-10 : 129*4+43*2+10]
        data_x6 = data_x0[:, :, 129*5+43-10 : 129*5+43*2+10]
        data_x7 = data_x0[:, :, 129*6+43-10 : 129*6+43*2+10]
        data_x8 = data_x0[:, :, 129*7+43-10 : 129*7+43*2+10]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_final.shape)
        inputsize = 63*8
    return data_final,inputsize

def readfile_fenpi_output_43yipi_verify(h5f_file, frequency, type_name ):
    if frequency == 0:
        data_x0 = np.array(h5f_file[type_name])
        # print('data000',data_x0.shape)
        data_x1 = data_x0[:, :, 0+129*0: 43+129*0]
        data_x2 = data_x0[:, :, 0+129*1: 43+129*1]
        data_x3 = data_x0[:, :, 0+129*2: 43+129*2]
        data_x4 = data_x0[:, :, 0+129*3: 43+129*3]
        data_x5 = data_x0[:, :, 0+129*4: 43+129*4]
        data_x6 = data_x0[:, :, 0+129*5: 43+129*5]
        data_x7 = data_x0[:, :, 0+129*6: 43+129*6]
        data_x8 = data_x0[:, :, 0+129*7: 43+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
    elif frequency == 2:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 129 * 1 - 43: 129 * 1]
        data_x2 = data_x0[:, :, 129 * 2 - 43: 129 * 2]
        data_x3 = data_x0[:, :, 129 * 3 - 43: 129 * 3]
        data_x4 = data_x0[:, :, 129 * 4 - 43: 129 * 4]
        data_x5 = data_x0[:, :, 129 * 5 - 43: 129 * 5]
        data_x6 = data_x0[:, :, 129 * 6 - 43: 129 * 6]
        data_x7 = data_x0[:, :, 129 * 7 - 43: 129 * 7]
        data_x8 = data_x0[:, :, 129 * 8 - 43: 129 * 8]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
        inputsize = 53 * 8
    else:
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 43+129*0 : 129*1-43]
        data_x2 = data_x0[:, :, 43+129*1 : 129*2-43]
        data_x3 = data_x0[:, :, 43+129*2 : 129*3-43]
        data_x4 = data_x0[:, :, 43+129*3 : 129*4-43]
        data_x5 = data_x0[:, :, 43+129*4 : 129*5-43]
        data_x6 = data_x0[:, :, 43+129*5 : 129*6-43]
        data_x7 = data_x0[:, :, 43+129*6 : 129*7-43]
        data_x8 = data_x0[:, :, 43+129*7 : 129*8-43]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape', data_final.shape)
    return data_final

def readfile_fenpi_input_43yipi_buduo(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0:43+129*0]
            data_x2 = data_x0[:, :, 0+129*1:43+129*1]
            data_x3 = data_x0[:, :, 0+129*2:43+129*2]
            data_x4 = data_x0[:, :, 0+129*3:43+129*3]
            data_x5 = data_x0[:, :, 0+129*4:43+129*4]
            data_x6 = data_x0[:, :, 0+129*5:43+129*5]
            data_x7 = data_x0[:, :, 0+129*6:43+129*6]
            data_x8 = data_x0[:, :, 0+129*7:43+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 43 * 8
        elif  frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*1-43 : 129*1]
            data_x2 = data_x0[:, :, 129*2-43 : 129*2]
            data_x3 = data_x0[:, :, 129*3-43 : 129*3]
            data_x4 = data_x0[:, :, 129*4-43 : 129*4]
            data_x5 = data_x0[:, :, 129*5-43 : 129*5]
            data_x6 = data_x0[:, :, 129*6-43 : 129*6]
            data_x7 = data_x0[:, :, 129*7-43 : 129*7]
            data_x8 = data_x0[:, :, 129*8-43 : 129*8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 43 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*0+43 : 129*0+43*2]
            data_x2 = data_x0[:, :, 129*1+43 : 129*1+43*2]
            data_x3 = data_x0[:, :, 129*2+43 : 129*2+43*2]
            data_x4 = data_x0[:, :, 129*3+43 : 129*3+43*2]
            data_x5 = data_x0[:, :, 129*4+43 : 129*4+43*2]
            data_x6 = data_x0[:, :, 129*5+43 : 129*5+43*2]
            data_x7 = data_x0[:, :, 129*6+43 : 129*6+43*2]
            data_x8 = data_x0[:, :, 129*7+43 : 129*7+43*2]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 43*8
    return data_total,inputsize

def readfile_fenpi_output_43yipi_buduo(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0: 43+129*0]
            data_x2 = data_x0[:, :, 0+129*1: 43+129*1]
            data_x3 = data_x0[:, :, 0+129*2: 43+129*2]
            data_x4 = data_x0[:, :, 0+129*3: 43+129*3]
            data_x5 = data_x0[:, :, 0+129*4: 43+129*4]
            data_x6 = data_x0[:, :, 0+129*5: 43+129*5]
            data_x7 = data_x0[:, :, 0+129*6: 43+129*6]
            data_x8 = data_x0[:, :, 0+129*7: 43+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
        elif frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129 * 1 - 43: 129 * 1]
            data_x2 = data_x0[:, :, 129 * 2 - 43: 129 * 2]
            data_x3 = data_x0[:, :, 129 * 3 - 43: 129 * 3]
            data_x4 = data_x0[:, :, 129 * 4 - 43: 129 * 4]
            data_x5 = data_x0[:, :, 129 * 5 - 43: 129 * 5]
            data_x6 = data_x0[:, :, 129 * 6 - 43: 129 * 6]
            data_x7 = data_x0[:, :, 129 * 7 - 43: 129 * 7]
            data_x8 = data_x0[:, :, 129 * 8 - 43: 129 * 8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
            inputsize = 53 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 43+129*0 : 129*1-43]
            data_x2 = data_x0[:, :, 43+129*1 : 129*2-43]
            data_x3 = data_x0[:, :, 43+129*2 : 129*3-43]
            data_x4 = data_x0[:, :, 43+129*3 : 129*4-43]
            data_x5 = data_x0[:, :, 43+129*4 : 129*5-43]
            data_x6 = data_x0[:, :, 43+129*5 : 129*6-43]
            data_x7 = data_x0[:, :, 43+129*6 : 129*7-43]
            data_x8 = data_x0[:, :, 43+129*7 : 129*8-43]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
    return data_total


def readfile_fenpi_input_43yipi(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0:43+10+129*0]
            data_x2 = data_x0[:, :, 0+129*1:43+10+129*1]
            data_x3 = data_x0[:, :, 0+129*2:43+10+129*2]
            data_x4 = data_x0[:, :, 0+129*3:43+10+129*3]
            data_x5 = data_x0[:, :, 0+129*4:43+10+129*4]
            data_x6 = data_x0[:, :, 0+129*5:43+10+129*5]
            data_x7 = data_x0[:, :, 0+129*6:43+10+129*6]
            data_x8 = data_x0[:, :, 0+129*7:43+10+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 53 * 8
        elif  frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*1-53 : 129*1]
            data_x2 = data_x0[:, :, 129*2-53 : 129*2]
            data_x3 = data_x0[:, :, 129*3-53 : 129*3]
            data_x4 = data_x0[:, :, 129*4-53 : 129*4]
            data_x5 = data_x0[:, :, 129*5-53 : 129*5]
            data_x6 = data_x0[:, :, 129*6-53 : 129*6]
            data_x7 = data_x0[:, :, 129*7-53 : 129*7]
            data_x8 = data_x0[:, :, 129*8-53 : 129*8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 53 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129*0+43-10 : 129*0+43*2+10]
            data_x2 = data_x0[:, :, 129*1+43-10 : 129*1+43*2+10]
            data_x3 = data_x0[:, :, 129*2+43-10 : 129*2+43*2+10]
            data_x4 = data_x0[:, :, 129*3+43-10 : 129*3+43*2+10]
            data_x5 = data_x0[:, :, 129*4+43-10 : 129*4+43*2+10]
            data_x6 = data_x0[:, :, 129*5+43-10 : 129*5+43*2+10]
            data_x7 = data_x0[:, :, 129*6+43-10 : 129*6+43*2+10]
            data_x8 = data_x0[:, :, 129*7+43-10 : 129*7+43*2+10]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 63*8
    return data_total,inputsize

def readfile_fenpi_output_43yipi(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency == 0:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0: 43+129*0]
            data_x2 = data_x0[:, :, 0+129*1: 43+129*1]
            data_x3 = data_x0[:, :, 0+129*2: 43+129*2]
            data_x4 = data_x0[:, :, 0+129*3: 43+129*3]
            data_x5 = data_x0[:, :, 0+129*4: 43+129*4]
            data_x6 = data_x0[:, :, 0+129*5: 43+129*5]
            data_x7 = data_x0[:, :, 0+129*6: 43+129*6]
            data_x8 = data_x0[:, :, 0+129*7: 43+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
        elif frequency == 2:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 129 * 1 - 43: 129 * 1]
            data_x2 = data_x0[:, :, 129 * 2 - 43: 129 * 2]
            data_x3 = data_x0[:, :, 129 * 3 - 43: 129 * 3]
            data_x4 = data_x0[:, :, 129 * 4 - 43: 129 * 4]
            data_x5 = data_x0[:, :, 129 * 5 - 43: 129 * 5]
            data_x6 = data_x0[:, :, 129 * 6 - 43: 129 * 6]
            data_x7 = data_x0[:, :, 129 * 7 - 43: 129 * 7]
            data_x8 = data_x0[:, :, 129 * 8 - 43: 129 * 8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
            inputsize = 53 * 8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, 43+129*0 : 129*1-43]
            data_x2 = data_x0[:, :, 43+129*1 : 129*2-43]
            data_x3 = data_x0[:, :, 43+129*2 : 129*3-43]
            data_x4 = data_x0[:, :, 43+129*3 : 129*4-43]
            data_x5 = data_x0[:, :, 43+129*4 : 129*5-43]
            data_x6 = data_x0[:, :, 43+129*5 : 129*6-43]
            data_x7 = data_x0[:, :, 43+129*6 : 129*7-43]
            data_x8 = data_x0[:, :, 43+129*7 : 129*8-43]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape', data_total.shape)
    return data_total







def readfile_fenpi_input(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        if frequency <= 9:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            # print('data000',data_x0.shape)
            data_x1 = data_x0[:, :, 0+129*0:frequency+11+129*0]
            data_x2 = data_x0[:, :, 0+129*1:frequency+11+129*1]
            data_x3 = data_x0[:, :, 0+129*2:frequency+11+129*2]
            data_x4 = data_x0[:, :, 0+129*3:frequency+11+129*3]
            data_x5 = data_x0[:, :, 0+129*4:frequency+11+129*4]
            data_x6 = data_x0[:, :, 0+129*5:frequency+11+129*5]
            data_x7 = data_x0[:, :, 0+129*6:frequency+11+129*6]
            data_x8 = data_x0[:, :, 0+129*7:frequency+11+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = (frequency + 11)*8
        elif  frequency >= 119:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, frequency-10+129*0 : 129*1]
            data_x2 = data_x0[:, :, frequency-10+129*1 : 129*2]
            data_x3 = data_x0[:, :, frequency-10+129*2 : 129*3]
            data_x4 = data_x0[:, :, frequency-10+129*3 : 129*4]
            data_x5 = data_x0[:, :, frequency-10+129*4 : 129*5]
            data_x6 = data_x0[:, :, frequency-10+129*5 : 129*6]
            data_x7 = data_x0[:, :, frequency-10+129*6 : 129*7]
            data_x8 = data_x0[:, :, frequency-10+129*7 : 129*8]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = (129 - frequency +10)*8
        else:
            dirs = path_2 + i
            h5f_file = h5py.File(dirs, 'r')
            data_x0 = np.array(h5f_file[type_name])
            data_x1 = data_x0[:, :, frequency-10+129*0 : frequency+11+129*0]
            data_x2 = data_x0[:, :, frequency-10+129*1 : frequency+11+129*1]
            data_x3 = data_x0[:, :, frequency-10+129*2 : frequency+11+129*2]
            data_x4 = data_x0[:, :, frequency-10+129*3 : frequency+11+129*3]
            data_x5 = data_x0[:, :, frequency-10+129*4 : frequency+11+129*4]
            data_x6 = data_x0[:, :, frequency-10+129*5 : frequency+11+129*5]
            data_x7 = data_x0[:, :, frequency-10+129*6 : frequency+11+129*6]
            data_x8 = data_x0[:, :, frequency-10+129*7 : frequency+11+129*7]
            data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
            if num_list == 0:
                data_total = data_final
            else:
                data_total = np.concatenate((data_total, data_final), axis=0)
            num_list = num_list + 1
            # print(i)
            # print('data_shape',data_final.shape)
            print('data_total_shape',data_total.shape)
            inputsize = 21*8
    return data_total,inputsize

def readfile_fenpi_output(list_title,path_2,frequency,type_name):
    num_list = 0
    for i in list_title:
        dirs = path_2 + i
        h5f_file = h5py.File(dirs, 'r')
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = np.expand_dims(data_x0[:, :, frequency + 129 * 0], axis=2)
        data_x2 = np.expand_dims(data_x0[:, :, frequency + 129 * 1], axis=2)
        data_x3 = np.expand_dims(data_x0[:, :, frequency + 129 * 2], axis=2)
        data_x4 = np.expand_dims(data_x0[:, :, frequency + 129 * 3], axis=2)
        data_x5 = np.expand_dims(data_x0[:, :, frequency + 129 * 4], axis=2)
        data_x6 = np.expand_dims(data_x0[:, :, frequency + 129 * 5], axis=2)
        data_x7 = np.expand_dims(data_x0[:, :, frequency + 129 * 6], axis=2)
        data_x8 = np.expand_dims(data_x0[:, :, frequency + 129 * 7], axis=2)
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        if num_list == 0:
            data_total = data_final
        else:
            data_total = np.concatenate((data_total, data_final), axis=0)
        num_list = num_list + 1
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_total.shape)
    return data_total

def readfile_fenpi_input_verify(filename,frequency,type_name):
    num_list = 0
    if frequency <= 9:
        h5f_file = filename
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, 0+129*0:frequency+11+129*0]
        data_x2 = data_x0[:, :, 0+129*1:frequency+11+129*1]
        data_x3 = data_x0[:, :, 0+129*2:frequency+11+129*2]
        data_x4 = data_x0[:, :, 0+129*3:frequency+11+129*3]
        data_x5 = data_x0[:, :, 0+129*4:frequency+11+129*4]
        data_x6 = data_x0[:, :, 0+129*5:frequency+11+129*5]
        data_x7 = data_x0[:, :, 0+129*6:frequency+11+129*6]
        data_x8 = data_x0[:, :, 0+129*7:frequency+11+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        print('data_shape',data_final.shape)
        inputsize = (frequency + 11)*8
    elif  frequency >= 119:
        h5f_file = filename
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, frequency-10+129*0 : 129*1]
        data_x2 = data_x0[:, :, frequency-10+129*1 : 129*2]
        data_x3 = data_x0[:, :, frequency-10+129*2 : 129*3]
        data_x4 = data_x0[:, :, frequency-10+129*3 : 129*4]
        data_x5 = data_x0[:, :, frequency-10+129*4 : 129*5]
        data_x6 = data_x0[:, :, frequency-10+129*5 : 129*6]
        data_x7 = data_x0[:, :, frequency-10+129*6 : 129*7]
        data_x8 = data_x0[:, :, frequency-10+129*7 : 129*8]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        print('data_shape',data_final.shape)
        inputsize = (129 - frequency +10)*8
    else:
        h5f_file = filename
        data_x0 = np.array(h5f_file[type_name])
        data_x1 = data_x0[:, :, frequency-10+129*0 : frequency+11+129*0]
        data_x2 = data_x0[:, :, frequency-10+129*1 : frequency+11+129*1]
        data_x3 = data_x0[:, :, frequency-10+129*2 : frequency+11+129*2]
        data_x4 = data_x0[:, :, frequency-10+129*3 : frequency+11+129*3]
        data_x5 = data_x0[:, :, frequency-10+129*4 : frequency+11+129*4]
        data_x6 = data_x0[:, :, frequency-10+129*5 : frequency+11+129*5]
        data_x7 = data_x0[:, :, frequency-10+129*6 : frequency+11+129*6]
        data_x8 = data_x0[:, :, frequency-10+129*7 : frequency+11+129*7]
        data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
        print('data_shape',data_final.shape)
        inputsize = 21*8
    return data_final,inputsize

def readfile_fenpi_output_verify(filename,frequency,type_name):
    h5f_file = filename
    data_x0 = np.array(h5f_file[type_name])
    data_x1 = np.expand_dims(data_x0[:, :, frequency + 129 * 0], axis=2)
    data_x2 = np.expand_dims(data_x0[:, :, frequency + 129 * 1], axis=2)
    data_x3 = np.expand_dims(data_x0[:, :, frequency + 129 * 2], axis=2)
    data_x4 = np.expand_dims(data_x0[:, :, frequency + 129 * 3], axis=2)
    data_x5 = np.expand_dims(data_x0[:, :, frequency + 129 * 4], axis=2)
    data_x6 = np.expand_dims(data_x0[:, :, frequency + 129 * 5], axis=2)
    data_x7 = np.expand_dims(data_x0[:, :, frequency + 129 * 6], axis=2)
    data_x8 = np.expand_dims(data_x0[:, :, frequency + 129 * 7], axis=2)
    data_final = np.concatenate((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7, data_x8),axis=2)
    print('data_shape',data_final.shape)
    return data_final