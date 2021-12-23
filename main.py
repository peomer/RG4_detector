import os

import tensorflow as tf
from keras.losses import mse
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from keras.layers import Dense , Conv2D , Flatten ,Conv1D , MaxPooling1D , MaxPool1D
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, LSTM, GRU, Bidirectional, Input, concatenate
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random as python_random
from tensorflow.keras.models import load_model
from csv import writer

def onehot_encoding(string):
    transtab = str.maketrans('ACGT','0123')
    string= str(string)
    data = [int(x) for x in list(string.translate(transtab))]
    almost = np.eye(4)[data]
    return almost

def tf_pearson(x, y):
    mx = tf.math.reduce_mean(input_tensor=x,keepdims=True)          # E[X]
    my = tf.math.reduce_mean(input_tensor=y,keepdims=True)          # E[Y]
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))    # E[(X-E[X])*(Y-E[Y])] = COV[X,Y]
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)         # sigma(X)*sigma(Y)
    return  r_num / r_den                                           # COV[X,Y] \ sigma(X)*sigma(Y)

def one_hot_enc(seq):
    seq = seq[:-1]
    seq = seq + "ACGT"
    if 'N' not in seq:
        trans = seq.maketrans('ACGT', '0123')
        numSeq = list(seq.translate(trans))
        return to_categorical(numSeq)[0:-4]
    else:
        trans = seq.maketrans('ACGTN', '01234')
        numSeq = list(seq.translate(trans))
        hotVec = to_categorical(numSeq)[0:-4]
        for i in range(len(hotVec)):
            if hotVec[i][4] == 1:
                hotVec[i] = [0.25,0.25,0.25,0.25,0]
        return np.delete(hotVec,4,1)

def trim_seq(array,how_much):
    halp_p = len(array[0][1])/2
    from_idx = round(halp_p-how_much/2)
    to_idx = round(halp_p+how_much/2)
    trim = array.apply([lambda x :x.str.slice(from_idx,to_idx)])
    return trim

# def get_data(path, min_read=2000,add_RNAplfold =False,export_np_arr=False,load_np_arr=False):
#     # train
#     if load_np_arr:
#         #Train
#         X_train = np.load(path+"/np_data/X_train_np.npy")
#         y_train = np.load(path+"/np_data/y_train_np.npy")
#         w_train = np.load(path + "/np_data/w_train_np.npy")
#         # Test
#         X_test = np.load(path + "/np_data/X_test_np.npy")
#         y_test = np.load(path + "/np_data/y_test_np.npy")
#         w_test = np.load(path + "/np_data/w_test_np.npy")
#         # Val
#         X_val = np.load(path + "/np_data/X_val_np.npy")
#         y_val = np.load(path + "/np_data/y_val_np.npy")
#         w_val = np.load(path + "/np_data/w_val_np.npy")
#
#     else:
#         with open(path+  "/seq/train-seq") as source:
#             X_train = np.array(list(map(one_hot_enc, source)))
#         y_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['rsr']).to_numpy()
#         w_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['c_read']).to_numpy() + \
#               pd.read_csv(path + '/csv_data/train_data.csv', usecols=['t_read']).to_numpy()
#         chr_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['chromosome']).to_numpy()
#         pos_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['position']).to_numpy()
#         strand_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['strand']).to_numpy()
#         # validation
#         with open(path+  "/seq/val-seq") as source:
#             X_val =  np.array(list(map(one_hot_enc, source)))
#         y_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['rsr']).to_numpy()
#         w_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['c_read']).to_numpy() +\
#             pd.read_csv(path + '/csv_data/val_data.csv', usecols=['t_read']).to_numpy()
#         chr_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['chromosome']).to_numpy()
#         pos_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['position']).to_numpy()
#         strand_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['strand']).to_numpy()
#         # test
#         with open(path+  "/seq/test-seq") as source:
#             X_test = np.array(list(map(one_hot_enc, source)))
#         y_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['rsr']).to_numpy()
#         w_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['c_read']).to_numpy() + \
#                  pd.read_csv(path + '/csv_data/test_data.csv', usecols=['t_read']).to_numpy()
#         chr_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['chromosome']).to_numpy()
#         pos_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['position']).to_numpy()
#         strand_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['strand']).to_numpy()
#
#         # set val min read
#         ids = np.argwhere(w_val > min_read)[:, 0]
#         X_val = X_val[ids]
#         y_val = y_val[ids]
#         w_val = w_val[ids]
#
#         ids = np.argwhere(w_test > min_read)[:, 0]
#         X_test = X_test[ids]
#         y_test = y_test[ids]
#         w_test = w_test[ids]
#         # scale_labels
#         y_train = np.log(y_train)
#         y_test = np.log(y_test)
#         y_val = np.log(y_val)
#
#     if (add_RNAplfold):
#         X_new = []
#         for i in range(len(X_train)):
#             plfold_name = ""
#             a = str(chr_train[i])[2:-2]
#             b = str(pos_train[i] - 140)[1:-1]
#             c = str(pos_train[i] + 110)[1:-1]
#             d = str(strand_train[i])[2:-2]
#             e = ')_lunp\_clean'
#             plfold_name = a + '_' + b + '-' + c + '(' + d + e
#             with open(path + "/plfold/train_plfold/" + plfold_name) as source:
#                 pl_train = np.array(list(source))
#             pl_train = pl_train.astype(float)
#             #pl_train1 = pd.read_csv(path + '/train_plfold/' + plfold_name).to_numpy()
#             temp = X_train[i]
#             temp = np.column_stack((temp, pl_train))
#             X_new.append(temp)
#         X_train = np.array(X_new)
#         X_new = []
#         for i in range(len(X_val)):
#             plfold_name = ""
#             a = str(chr_val[i])[2:-2]
#             b = str(pos_val[i] - 140)[1:-1]
#             c = str(pos_val[i] + 110)[1:-1]
#             d = str(strand_val[i])[2:-2]
#             e = ')_lunp\_clean'
#             plfold_name = a + '_' + b + '-' + c + '(' + d + e
#             with open(path + "/plfold/val_plfold/" + plfold_name) as source:
#                 pl_val = np.array(list(source))
#             pl_val = pl_val.astype(float)
#             #pl_train = pd.read_csv(path + '/val_plfold/' + plfold_name).to_numpy()
#             temp = X_val[i]
#             temp = np.column_stack((temp, pl_val))
#             X_new.append(temp)
#         X_val = np.array(X_new)
#         X_new = []
#         for i in range(len(X_test)):
#             plfold_name = ""
#             a = str(chr_test[i])[2:-2]
#             b = str(pos_test[i] - 140)[1:-1]
#             c = str(pos_test[i] + 110)[1:-1]
#             d = str(strand_test[i])[2:-2]
#             e = ')_lunp\_clean'
#             plfold_name = a + '_' + b + '-' + c + '(' + d + e
#             with open(path + "/plfold/test_plfold/" + plfold_name) as source:
#                 pl_test = np.array(list(source))
#             pl_test = pl_test.astype(float)
#             #pl_train = pd.read_csv(path + '/test_plfold/' + plfold_name).to_numpy()
#             temp = X_test[i]
#             temp = np.column_stack((temp, pl_test))
#             X_new.append(temp)
#         X_test = np.array(X_new)
#
#
#     if export_np_arr :
#         np.save(path + "/np_data/X_train_np.npy",X_train)
#         np.save(path + "/np_data/y_train_np", y_train)
#         np.save(path + "/np_data/w_train_np", w_train)
#
#         np.save(path + "/np_data/X_test_np", X_test)
#         np.save(path + "/np_data/y_test_np", y_test)
#         np.save(path + "/np_data/w_test_np", w_test)
#
#         np.save(path + "/np_data/X_val_np", X_val)
#         np.save(path + "/np_data/y_val_np", y_val)
#         np.save(path + "/np_data/w_val_np", w_val)
#
#     return [X_train, y_train, w_train], [X_test, y_test, w_test], [X_val, y_val, w_val]
def get_data(path, min_read=2000,add_RNAplfold =False,export_np_arr=False,load_np_arr=False,add_evulution=False,add_mfe=False):
    # train
    if load_np_arr:
        folder = "/np_data"

        if add_RNAplfold:
            folder = "/np_data_pl"

        if add_evulution:
            folder = "/np_data_evu"

        if add_mfe :
            folder = "/np_data_mfe"


        #Train
        X_train = np.load(path+ folder + "/X_train.npy")
        y_train = np.load(path+ folder + "/y_train.npy")
        w_train = np.load(path + folder + "/w_train.npy")
        if add_mfe:
            mfe_train = np.load(path + folder + "/mfe_train.npy")

        # Test
        X_test = np.load(path + folder + "/X_test.npy")
        y_test = np.load(path + folder + "/y_test.npy")
        w_test = np.load(path + folder + "/w_test.npy")
        if add_mfe:
            mfe_test = np.load(path + folder + "/mfe_test.npy")

        # Val
        X_val = np.load(path + folder + "/X_val.npy")
        y_val = np.load(path + folder + "/y_val.npy")
        w_val = np.load(path + folder + "/w_val.npy")
        if add_mfe:
            mfe_val = np.load(path + folder + "/mfe_val.npy")

    else:
        with open(path+  "/seq/train-seq") as source:
            X_train = np.array(list(map(one_hot_enc, source)))
        y_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['rsr']).to_numpy()
        w_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['c_read']).to_numpy() + \
              pd.read_csv(path + '/csv_data/train_data.csv', usecols=['t_read']).to_numpy()
        chr_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['chromosome']).to_numpy()
        pos_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['position']).to_numpy()
        pos_train = pos_train.astype(float)
        strand_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['strand']).to_numpy()
        if add_mfe:
            mfe_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['mfe']).to_numpy()
        # validation
        with open(path+  "/seq/val-seq") as source:
            X_val =  np.array(list(map(one_hot_enc, source)))
        y_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['rsr']).to_numpy()
        w_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['c_read']).to_numpy() +\
            pd.read_csv(path + '/csv_data/val_data.csv', usecols=['t_read']).to_numpy()
        chr_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['chromosome']).to_numpy()
        pos_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['position']).to_numpy()
        pos_val = pos_val.astype(float)
        strand_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['strand']).to_numpy()
        if add_mfe:
            mfe_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['mfe']).to_numpy()
        # test
        with open(path+  "/seq/test-seq") as source:
            X_test = np.array(list(map(one_hot_enc, source)))
        y_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['rsr']).to_numpy()
        w_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['c_read']).to_numpy() + \
                 pd.read_csv(path + '/csv_data/test_data.csv', usecols=['t_read']).to_numpy()
        chr_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['chromosome']).to_numpy()
        pos_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['position']).to_numpy()
        pos_test = pos_test.astype(float)
        strand_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['strand']).to_numpy()
        if add_mfe:
            mfe_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['mfe']).to_numpy()

        # set val min read
        ids = np.argwhere(w_val > min_read)[:, 0]
        X_val = X_val[ids]
        y_val = y_val[ids]
        w_val = w_val[ids]
        if add_mfe:
            mfe_val = mfe_val[ids]
        #print(len(ids))
        # set test min read
        ids = np.argwhere(w_test > min_read)[:, 0]
        X_test = X_test[ids]
        y_test = y_test[ids]
        w_test = w_test[ids]
        if add_mfe:
            mfe_test = mfe_test[ids]
        #print(len(ids))
        # scale_labels
        y_train = np.log(y_train)
        y_test = np.log(y_test)
        y_val = np.log(y_val)

        if add_evulution:
            ref_dic = {}
            values = {}
            for i in range(23):
                if i == 0:
                    chrom = 'chrX'
                else:
                    chrom = 'chr' + str(i)
                file_evu = chrom + '.phastCons100way.wigFix'
                head_file = chrom + '_fixed_nums'
                with open('./evulutionary_conservation/' + file_evu) as source:
                    values_temp = list(source)
                values_temp = np.array(values_temp)
                with open('./evulutionary_conservation/' + head_file) as source:
                    head_nums = np.array(list(source))
                head_nums = head_nums.astype(int)
                idx = 0
                #values_temp1 = np.array([])
                #values_temp1 = []
                values_temp1 = np.zeros(1000000000)
                ref_dic[chrom] = []
                time_list1 = []
                time_list2 = []
                idx=0
                for f in range(len(head_nums)):
                    temp_time = time.time()
                    head_nums[f] = head_nums[f]-1
                    headline = values_temp[head_nums[f]]
                    ref_dic[chrom].append(get_head(headline))
                    if f==0:
                        continue
                    #temp = np.concatenate((values_temp[head_nums[f-1]+1:head_nums[f]],np.zeros(ref_dic[chrom][-1]-ref_dic[chrom][-2]-head_nums[f]+head_nums[f-1]+1)))
                    a = ref_dic[chrom][-1]-ref_dic[chrom][-2]-head_nums[f]+head_nums[f-1]+1
                    #values_temp1 = np.concatenate((values_temp1,values_temp[head_nums[f-1]+1:head_nums[f]]))
                    #a=[ref_dic[chrom][-1] , ref_dic[chrom][-2] ,head_nums[f] , head_nums[f - 1]]
                    #values_temp1 = np.concatenate((values_temp1,np.zeros(ref_dic[chrom][-1]-ref_dic[chrom][-2]-head_nums[f]+head_nums[f-1]+1)))
                    #values_temp1 = np.concatenate((values_temp1,temp))
                    #values_temp1.append(values_temp1,temp)
                    start = head_nums[f - 1] + 1
                    end = head_nums[f]
                    values_temp1[idx:idx+end-start] = values_temp[start:end]
                    idx = idx+end-start+a
                    time_list2.append(time.time()-temp_time)
                print(time_list2)
                #values_temp = np.array(values_temp[1:])
                #values_temp1 = np.concatenate((values_temp1, values_temp[head_nums[f - 1]:]))
                start = head_nums[f] + 1
                end = len(values_temp)
                values_temp1[idx:idx + end - start] = values_temp[head_nums[f]+1:]
                idx = idx + end - start
                values[chrom] = values_temp1[:idx].astype(float)

                np.save("./rg4_data/np_data_evu/" + chrom + ".npy", values[chrom])
            X_new = []
            for i in range(len(X_train)):
                chrom = str(chr_train[i])[2:-2]
                start = pos_train[i] - 140 - ref_dic[chrom][0]
                end = pos_train[i] + 110 - ref_dic[chrom][0]
                temp = X_train[i]
                temp = np.column_stack((temp, values[chrom][start:end]))
                X_new.append(temp)
            X_train = np.array(X_new)
            X_new = []
            for i in range(len(X_test)):
                chrom = str(chr_test[i])[2:-2]
                start = pos_test[i] - 140 - ref_dic[chrom][0]
                end = pos_test[i] + 110 - ref_dic[chrom][0]
                temp = X_test[i]
                temp = np.column_stack((temp, values[chrom][start:end]))
                X_new.append(temp)
            X_test = np.array(X_new)
            X_new = []
            for i in range(len(X_val)):
                chrom = str(chr_val[i])[2:-2]
                start = pos_val[i] - 140 - ref_dic[chrom][0]
                end = pos_val[i] + 110 - ref_dic[chrom][0]
                temp = X_val[i]
                temp = np.column_stack((temp, values[chrom][start:end]))
                X_new.append(temp)
            X_val = np.array(X_new)

        if (add_RNAplfold):
            X_new = []
            for i in range(len(X_train)):
                plfold_name = ""
                a = str(chr_train[i])[2:-2]
                b = str(pos_train[i] - 140)
                c = str(pos_train[i] + 110)
                d = str(strand_train[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/train_plfold/" + plfold_name) as source:
                    pl_train = np.array(list(source))
                pl_train = pl_train.astype(float)
                #pl_train1 = pd.read_csv(path + '/train_plfold/' + plfold_name).to_numpy()
                temp = X_train[i]
                temp = np.column_stack((temp, pl_train))
                X_new.append(temp)
            X_train = np.array(X_new)
            X_new = []
            for i in range(len(X_val)):
                plfold_name = ""
                a = str(chr_val[i])[2:-2]
                b = str(pos_val[i] - 140)
                c = str(pos_val[i] + 110)
                d = str(strand_val[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/val_plfold/" + plfold_name) as source:
                    pl_val = np.array(list(source))
                pl_val = pl_val.astype(float)
                #pl_train = pd.read_csv(path + '/val_plfold/' + plfold_name).to_numpy()
                temp = X_val[i]
                temp = np.column_stack((temp, pl_val))
                X_new.append(temp)
            X_val = np.array(X_new)
            X_new = []
            for i in range(len(X_test)):
                plfold_name = ""
                a = str(chr_test[i])[2:-2]
                b = str(pos_test[i] - 140)
                c = str(pos_test[i] + 110)
                d = str(strand_test[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/test_plfold/" + plfold_name) as source:
                    pl_test = np.array(list(source))
                pl_test = pl_test.astype(float)
                #pl_train = pd.read_csv(path + '/test_plfold/' + plfold_name).to_numpy()
                temp = X_test[i]
                temp = np.column_stack((temp, pl_test))
                X_new.append(temp)
            X_test = np.array(X_new)


    if export_np_arr :

        folder = "/np_data"
        if add_RNAplfold:
            folder = "/np_data_pl"
        if add_evulution:
            folder = "/np_data_evu"
        if add_mfe :
            folder = "/np_data_mfe"


        np.save(path + folder + "/X_train.npy",X_train)
        np.save(path + folder + "/y_train.npy", y_train)
        np.save(path + folder + "/w_train.npy", w_train)


        np.save(path + folder + "/X_test.npy", X_test)
        np.save(path + folder + "/y_test.npy", y_test)
        np.save(path + folder + "/w_test.npy", w_test)


        np.save(path + folder + "/X_val.npy", X_val)
        np.save(path + folder + "/y_val.npy", y_val)
        np.save(path + folder + "/w_val.npy", w_val)

        if add_mfe:
            np.save(path + folder + "/mfe_val.npy", mfe_val)
            np.save(path + folder + "/mfe_test.npy", mfe_test)
            np.save(path + folder + "/mfe_train.npy", mfe_train)

    if add_mfe :
        return [X_train, y_train, w_train, mfe_train], [X_test, y_test, w_test, mfe_test], [X_val, y_val, w_val, mfe_val]
    else :
        return [X_train, y_train, w_train], [X_test, y_test, w_test], [X_val, y_val, w_val]

def get_head(headline):
    for j in range(len(headline)):
        if (headline[j] == '=' and headline[j - 5:j] == 'start'):
            h = headline.find(' ', j)
            ref_num = headline[j + 1:h]
            break
    return (int(ref_num))

def trim_mat(data,INPUT_SIZE,mfe=False):
    if mfe:
        [X_Data, Y_Data, W_Data, mfe_Data] = data
    else :
        [X_Data, Y_Data, W_Data] = data
    total_data_size = X_Data.shape[1]
    start = total_data_size // 2 - INPUT_SIZE // 2
    end = start + INPUT_SIZE
    X_Data = X_Data[:, start:end, :]
    if mfe :
        data = [X_Data, Y_Data, W_Data, mfe_Data]
    else:
        data = [X_Data, Y_Data, W_Data]
    return data

    # [X_Data, Y_Data, W_Data] = data
    # total_data_size = X_Data.shape[1]
    # start = total_data_size // 2 - INPUT_SIZE // 2
    # end = start + INPUT_SIZE
    # X_Data = X_Data[:, start:end, :]
    # data = [X_Data,Y_Data,W_Data]
    # return data

class HyperParams:

    def __init__(self):
        self.input_size_list = [60,80,100,120]
        self.conv_size_list = [16, 32, 64, 128]
        self.dense_size_list = [16, 32, 64, 128]
        self.dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lr_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.activations_list = ['relu', 'sigmoid', 'relu']
        self.batch_size_list = [16, 32, 64, 128]
        self.epochs_list = [x for x in range(1, 10)]
        self.INPUT_SIZE = 100
        self.FILTER = 32
        self.KERNEL_SIZE = 42
        self.POOLING = 1
        self.POOL_SIZE = 4
        self.DENCE_1 = 64
        self.DENCE_2 = 16
        self.ACTIVATION_1 = 'relu'
        self.ACTIVATION_2 = 'relu'
        self.DROPOUT_1 =0.3
        self.DROPOUT_2 = 0.1
        self.TF_SEED = 1
        self.EPOCH = 10
        self.BATCH_SIZE = 64
        self.CONV_PADDING = "valid"
        self.path = "./hparams/hyper_params.csv"

    def rand_params(self):
        self.INPUT_SIZE    = np.random.choice(self.input_size_list)
        self.FILTER        = np.random.randint(8, 12) * 8
        self.KERNEL_SIZE   = np.random.randint(1, self.INPUT_SIZE / 16 + 1) * 4
        self.POOLING       = 1
        self.POOL_SIZE     = np.random.randint(1, self.INPUT_SIZE / 16 + 1) * 4
        self.DENCE_1       = np.random.randint(6, 16) * 4
        self.DENCE_2       = np.random.randint(4, 16) * 4
        self.ACTIVATION_1  = np.random.choice(self.activations_list)
        self.ACTIVATION_2  = 'relu'
        self.DROPOUT_1     = np.random.randint(0, 8) * 0.1
        self.DROPOUT_2     = np.random.randint(0, 8) * 0.1
        self.TF_SEED       = np.random.randint(10000)
        self.EPOCH         = np.random.randint(3, 30)
        self.BATCH_SIZE    = np.random.randint(1, 10) * 8

    def print(self):
        print("INPUT_SIZE = ", self.INPUT_SIZE)
        print("FILTER = ", self.FILTER)
        print("KERNEL_SIZE = ", self.KERNEL_SIZE)
        print("POOL_SIZE = ", self.POOL_SIZE)
        print("DENCE_1 = ", self.DENCE_1)
        print("DENCE_2 = ", self.DENCE_2)
        print("ACTIVATION_1 = ", self.ACTIVATION_1)
        print("ACTIVATION_2 = ", self.ACTIVATION_2)
        print("DROPOUT_1 = ", self.DROPOUT_1)
        print("DROPOUT_2 = ", self.DROPOUT_2)
        print("TF_SEED = ", self.TF_SEED)
        print("EPOCH = ", self.EPOCH)
        print("BATCH_SIZE = ", self.BATCH_SIZE)
    def load_params(self,max=True,idx=0):
        Pearson_cor = pd.read_csv(self.path, usecols=['pearson correlation']).to_numpy()
        max_idx = np.argmax(Pearson_cor)
        INPUT_SIZE = pd.read_csv(self.path, usecols=['INPUT_SIZE']).to_numpy()
        FILTER = pd.read_csv(self.path, usecols=['FILTER']).to_numpy()
        KERNEL_SIZE = pd.read_csv(self.path, usecols=['KERNEL_SIZE']).to_numpy()
        POOL_SIZE = pd.read_csv(self.path, usecols=['POOL_SIZE']).to_numpy()
        DENCE_1 = pd.read_csv(self.path, usecols=['DENCE_1']).to_numpy()
        DENCE_2 = pd.read_csv(self.path, usecols=['DENCE_2']).to_numpy()
        ACTIVATION_1 = pd.read_csv(self.path, usecols=['ACTIVATION_1']).to_numpy()
        ACTIVATION_2 = pd.read_csv(self.path, usecols=['ACTIVATION_2']).to_numpy()
        DROPOUT_1 = pd.read_csv(self.path, usecols=['DROPOUT_1']).to_numpy()
        DROPOUT_2 = pd.read_csv(self.path, usecols=['DROPOUT_2']).to_numpy()
        TF_SEED = pd.read_csv(self.path, usecols=['TF_SEED']).to_numpy()
        EPOCH = pd.read_csv(self.path, usecols=['EPOCH']).to_numpy()
        BATCH_SIZE = pd.read_csv(self.path, usecols=['BATCH_SIZE']).to_numpy()

        read_idx = 0
        if max:
            read_idx =max_idx
        else:
            read_idx = idx
        self.INPUT_SIZE     = np.asscalar(INPUT_SIZE[read_idx])
        self.FILTER         = np.asscalar(FILTER[read_idx])
        self.KERNEL_SIZE    = np.asscalar(KERNEL_SIZE[read_idx])
        self.POOL_SIZE      = np.asscalar(POOL_SIZE[read_idx])
        self.DENCE_1        = np.asscalar(DENCE_1[read_idx])
        self.DENCE_2        = np.asscalar(DENCE_2[read_idx])
        self.ACTIVATION_1   = str((ACTIVATION_1[read_idx])[0])
        self.ACTIVATION_2   = str((ACTIVATION_2[read_idx])[0])
        self.DROPOUT_1      = np.asscalar(DROPOUT_1[read_idx])
        self.DROPOUT_2      = np.asscalar(DROPOUT_2[read_idx])
        self.TF_SEED        = np.asscalar(TF_SEED[read_idx])
        self.EPOCH          = np.asscalar(EPOCH[read_idx])
        self.BATCH_SIZE     =  np.asscalar(BATCH_SIZE[read_idx])

    def save_params(self):
        to_add = [self.INPUT_SIZE, self.FILTER,self.KERNEL_SIZE,self.POOL_SIZE,self.DENCE_1,self.DENCE_2,self.ACTIVATION_1,self.ACTIVATION_2,self.DROPOUT_1,self.DROPOUT_2,self.TF_SEED,self.EPOCH,self.BATCH_SIZE]
        with open(self.path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(to_add)
            f_object.close()

def build_mfe_model(hparams=HyperParams(),input_shape=None,mfe_shape=None):

    dataIn = Input(shape=input_shape)
    model1 = Conv1D(filters=hparams.FILTER, kernel_size=hparams.KERNEL_SIZE, input_shape=input_shape, name="conv", padding=hparams.CONV_PADDING)(dataIn)
    model1 = MaxPool1D(pool_size=hparams.POOL_SIZE, name="pooling")(model1)
    model1 = Dropout(hparams.DROPOUT_1)(model1)
    model1 = Flatten()(model1)
    mfeIn = Input(shape=mfe_shape)
    model2 = Flatten()(mfeIn)
    model = concatenate([model1, model2])
    model = Dense(hparams.DENCE_1, activation='relu', name="dense")(model)
    model = Dropout(hparams.DROPOUT_2)(model)
    model   = Dense(hparams.DENCE_2, activation='relu', name="dense2")(model)
    out     = Dense(1, activation='linear', name="1dense")(model)
    model = Model([dataIn, mfeIn], out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    if Debug:
        model.summary()
    return model

def build_seq_model(Load_model=False,hparams=HyperParams(),input_shape=()):
    if not Load_model:
        model = Sequential()
        model.add(Conv1D(filters=hparams.FILTER, kernel_size=hparams.KERNEL_SIZE, input_shape=input_shape,
                         name="conv", padding=hparams.CONV_PADDING))
        model.add(MaxPool1D(pool_size=hparams.POOL_SIZE, name="pooling"))
        model.add(Dropout(hparams.DROPOUT_1))
        model.add(Flatten())
        model.add(Dense(hparams.DENCE_1, activation=hparams.ACTIVATION_1, name="dense"))
        model.add(Dropout(hparams.DROPOUT_2))
        model.add(Dense(hparams.DENCE_2, activation=hparams.ACTIVATION_2, name="dense2"))
        # model.add(Dense(DENCE_1, activation='relu', name="dense3"))
        # model.add(Dense(32, activation='relu', name="dense4"))
        # model.add(Dense(16, activation='relu', name="dense5"))
        model.add(Dense(1, activation='linear', name="1dense"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        if Debug:
            model.summary()
    else:
        model = load_model('my_model')
    return model

def update_results_params(i=0,hparams=HyperParams()):

    Results_pd.at[i, 'INPUT_SIZE'] = hparams.INPUT_SIZE
    Results_pd.at[i, 'FILTER'] = hparams.FILTER
    Results_pd.at[i, 'KERNEL_SIZE'] = hparams.KERNEL_SIZE
    Results_pd.at[i, 'POOLING'] = hparams.POOLING
    Results_pd.at[i, 'POOL_SIZE'] = hparams.POOL_SIZE
    Results_pd.at[i, 'DENCE_1'] = hparams.DENCE_1
    Results_pd.at[i, 'DENCE_2'] = hparams.DENCE_2
    Results_pd.at[i, 'ACTIVATION_1'] = hparams.ACTIVATION_1
    Results_pd.at[i, 'ACTIVATION_2'] = hparams.ACTIVATION_2
    Results_pd.at[i, 'DROPOUT_1'] = hparams.DROPOUT_1
    Results_pd.at[i, 'DROPOUT_2'] = hparams.DROPOUT_2
    Results_pd.at[i, 'TF_SEED'] = hparams.TF_SEED
    Results_pd.at[i, 'EPOCH'] = hparams.EPOCH
    Results_pd.at[i, 'BATCH_SIZE'] = hparams.BATCH_SIZE
    Results_pd.at[i, 'CONV_PADDING'] = hparams.CONV_PADDING


def predict_results(model=Sequential(),X=None,Y=None):
    predictions = model.predict(X, batch_size=len(X))     # Batch_Size defualt is 32
    predictions = predictions.reshape(len(predictions))   # Reshape pred_val
    Y = Y.reshape(len(Y))                                 # Reshape Y_VAL
    [pearson,p_value] = pearsonr(Y,predictions)
    mse = np.mean(np.square(Y - predictions))
    return [pearson,p_value,mse]

def plot_loss_val(loss=None,val_loss=None,load_np_array=False):

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

#---------------------------------------------- End of functions def----------------------------------------------------






#---------------------------------------------- Configure Params -------------------------------------------------------

import params
Cloud_run = params.Cloud_run
global Debug
Debug= params.Debug
save_model = params.save_model
Load_model = params.save_model
ITER      = 3
param_scan = True
Seed_scan = False
compare_same_params = False
add_mfe = False
AVG_TF_SEED_AMOUNT = 3
Show_plots =False
Save_plots = True
Data_run_list = ['seq','mfe','evo','plfold']
# Data_run_list = ['mfe']
Save_Results = True
Use_Test = True

if Cloud_run :

    # path = "/home/u110379/RG4_Proj/rg4_data"
    path = "~/RG4_Proj/rg4_data"
    # path = "./rg4_data"
else:
    path = "./rg4_data"


if Debug :
    name = "debug"
    runidx = 0
    iterations = 1
    VERBOSE = 1
else :
    VERBOSE = 0
    name = "plfold_vs_nonpl_scan_round_2"
    runidx = input()
    iterations = ITER
#---------------------------------------------- Configure DataFrame ----------------------------------------------------
# Results Data frame :
# adding Hyper Parapaters
Results_pd = pd.DataFrame(
    columns=['INPUT_SIZE', 'FILTER', 'KERNEL_SIZE', 'POOLING', 'POOL_SIZE', 'DENCE_1', 'DENCE_2', 'ACTIVATION_1',
             'ACTIVATION_2', 'DROPOUT_1', 'DROPOUT_2', "TF_SEED", 'BATCH_SIZE', 'EPOCH', 'CONV_PADDING'])
# Result columns depending on the Data types to be Run.
witch_data_types = ''
for data_type in Data_run_list:
    witch_data_types = witch_data_types+ data_type + "_"
    Results_pd['loss_mse_val_' + data_type] = 'Nan'
    Results_pd['pearson_correlation_val_' + data_type] = 'Nan'
    Results_pd['p_value_val_' + data_type] ='Nan'

    if Use_Test:
        Results_pd['loss_mse_test_' + data_type] = 'Nan'
        Results_pd['pearson_correlation_test_' + data_type] = 'Nan'
        Results_pd['p_value_test_' + data_type] = 'Nan'

CSV_Path = "./out_results/"+"out_results_csv_"+name + "_" + witch_data_types  + "/"
os.makedirs(CSV_Path,exist_ok=True)
#---------------------------------------------- Read data---------------------------------------------------------------
if 'seq' in Data_run_list:
    train, test, validation = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False, add_evulution=False, add_mfe=False)

if 'mfe' in Data_run_list:
        train_mfe, test_mfe, validation_mfe  = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,add_evulution=False, add_mfe=True)

if 'evo' in Data_run_list:
        train_evo, test_evo, validation_evo = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,add_evulution=True, add_mfe=False)

if 'plfold' in Data_run_list:
        train_pl, test_pl, validation_pl = get_data(path, load_np_arr=True, add_RNAplfold=True, export_np_arr=True,add_evulution=False, add_mfe=False)




#---------------------------------------------- Main -------------------------------------------------------------------

def main(param_scan=True,Seed_scan=False):

        for i in range(ITER):       # Main Loop Runs ITER times
            hparams = HyperParams() # Init Hyper paramaters

            if param_scan:
                hparams.rand_params()

            if Seed_scan:
                hparams.load_params(max=True)              # Load parmas from CSV
                hparams.TF_SEED = np.random.randint(10000) # Randomize Seed

            if Debug :
                hparams.print()
            # add them to Pandas
            update_results_params(i,hparams)

            # Trim Data
            if 'seq' in Data_run_list:
                [X_train, Y_train, W_train]                     = trim_mat(train, hparams.INPUT_SIZE)
                [X_test, Y_test, W_test]                        = trim_mat(test, hparams.INPUT_SIZE)
                [X_validation, Y_validation, W_validation]      = trim_mat(validation, hparams.INPUT_SIZE)

            if 'mfe' in Data_run_list:
                [X_train_mfe, Y_train_mfe, W_train_mfe,mfe_train]              = trim_mat(train_mfe, hparams.INPUT_SIZE,mfe=True)
                [X_test_mfe, Y_test_mfe, W_test_mfe,mfe_test]                  = trim_mat(test_mfe, hparams.INPUT_SIZE,mfe=True)
                [X_validation_mfe, Y_validation_mfe, W_validation_mfe,mfe_val] = trim_mat(validation_mfe, hparams.INPUT_SIZE,mfe=True)

            if 'evo' in Data_run_list:
                [X_train_evo, Y_train_evo, W_train_evo] = trim_mat(train_evo, hparams.INPUT_SIZE)
                [X_test_evo, Y_test_evo, W_test_evo] = trim_mat(test_evo, hparams.INPUT_SIZE)
                [X_validation_evo, Y_validation_evo, W_validation_evo] = trim_mat(validation_evo, hparams.INPUT_SIZE)

            if 'plfold' in Data_run_list:
                [X_train_pl, Y_train_pl, W_train_pl]                = trim_mat(train_pl, hparams.INPUT_SIZE)
                [X_test_pl, Y_test_pl, W_test_pl]                   = trim_mat(test_pl, hparams.INPUT_SIZE)
                [X_validation_pl, Y_validation_pl, W_validation_pl] = trim_mat(validation_pl, hparams.INPUT_SIZE)


            for data_type in Data_run_list:
                # prepare the data to be trained on:
                if data_type == 'mfe':
                    X_TRAIN  = [X_train_mfe,mfe_train]
                    Y_TRAIN  = Y_train_mfe
                    X_VAL    = [X_validation_mfe,mfe_val]
                    Y_VAL    = Y_validation_mfe
                    X_TEST   = [X_test_mfe,mfe_test]
                    Y_TEST   = Y_test_mfe
                    mfe_shape = mfe_train.shape[1:]
                    data_shape = X_train_mfe.shape[1:]
                elif data_type == 'seq' :
                    X_TRAIN = X_train
                    Y_TRAIN = Y_train
                    X_VAL   = X_validation
                    Y_VAL   = Y_validation
                    X_TEST  = X_test
                    Y_TEST  =  Y_test
                    data_shape = X_train.shape[1:]
                elif data_type == 'evo' :
                    X_TRAIN = X_train_evo
                    Y_TRAIN = Y_train_evo
                    X_VAL   = X_validation_evo
                    Y_VAL   = Y_validation_evo
                    X_TEST  = X_test_evo
                    Y_TEST  = Y_test_evo
                    data_shape = X_train_evo.shape[1:]
                elif data_type == 'plfold':
                    X_TRAIN = X_train_pl
                    Y_TRAIN = Y_train_pl
                    X_VAL   = X_validation_pl
                    Y_VAL   = Y_validation_pl
                    X_TEST  = X_test_pl
                    Y_TEST  = Y_test_pl
                    data_shape = X_train_pl.shape[1:]

                # Reset Lists for Averging over TF Seeds
                TF_SEED_LIST          = []
                pearson_cor_list      = []
                p_value_list          = []
                Mse_list              = []
                loss_list             = []
                val_loss_list         = []
                pearson_cor_list_test = []
                p_value_list_test     = []
                Mse_list_test         = []
                for k in range(AVG_TF_SEED_AMOUNT):
                    # Generate a random seed
                    if k != 0 :
                        hparams.TF_SEED = np.random.randint(10000)

                    # Add Seed to List :
                    TF_SEED_LIST.append(hparams.TF_SEED)

                    #Set the TF Seed before the model build:

                    # The below is necessary for starting Numpy generated random numbers
                    # in a well-defined initial state.
                    # np.random.seed(TF_SEED)

                    # The below is necessary for starting core Python generated random numbers
                    # in a well-defined state.
                    # python_random.seed(TF_SEED)

                    # The below set_seed() will make random number generation
                    # in the TensorFlow backend have a well-defined initial state.
                    # For further details, see:  https://www.tensorflow.org/api_docs/python/tf/random/set_seed
                    tf.random.set_seed(hparams.TF_SEED)

                    if data_type == 'mfe' :
                        model = build_mfe_model(hparams=hparams,mfe_shape=mfe_shape,input_shape=data_shape)
                    else:
                        model = build_seq_model(hparams=hparams, Load_model=False, input_shape=data_shape)

                    if Debug:
                        hparams.EPOCH = 1

                    # Train Model

                    model.fit(x=X_TRAIN, y=Y_TRAIN, epochs=hparams.EPOCH, verbose=VERBOSE, batch_size=hparams.BATCH_SIZE,validation_data=(X_VAL,Y_VAL))

                    # Plot Loss for val and training
                    if Debug:
                        # summarize history for loss
                        loss_list.append(model.history.history['loss'])
                        val_loss_list.append(model.history.history['val_loss'])

                    # Predictions for Test
                    if Use_Test:
                        [pearson_cor, p_value, mse] = predict_results(model, X_TEST, Y_TEST)
                        pearson_cor_list_test.append(pearson_cor)
                        p_value_list_test.append(p_value)
                        Mse_list_test.append(mse)

                    # Predictions for Validation
                    [pearson_cor,p_value,mse] = predict_results(model,X_VAL,Y_VAL)
                    pearson_cor_list.append(pearson_cor)
                    p_value_list.append(p_value)
                    Mse_list.append(mse)



                # Average from all AVG_TF_SEED_PER Validation
                Avg_pearson  = np.average(pearson_cor_list)
                Avg_p_value  = np.average(p_value_list)
                Avg_mse      = np.average(Mse_list)
                Avg_loss     = np.average(loss_list,axis=0)         # Average done on Axis 0 meaning on all first elements etc
                Avg_val_loss = np.average(val_loss_list,axis=0)

                # Update results Data frame Validation :
                Results_pd.at[i, 'pearson_correlation_val_'+ data_type] = Avg_pearson
                Results_pd.at[i, 'p_value_val_' + data_type]            = Avg_p_value
                Results_pd.at[i, 'loss_mse_val_' +data_type]            = Avg_mse

                if Use_Test:
                    # Average from all AVG_TF_SEED_PER Test:
                    Avg_pearson_test = np.average(pearson_cor_list_test)
                    Avg_p_value_test = np.average(p_value_list_test)
                    Avg_mse_test = np.average(Mse_list_test)

                    # Update results Data frame Test :
                    Results_pd.at[i,'loss_mse_test_' + data_type] = Avg_mse_test
                    Results_pd.at[i,'pearson_correlation_test_' + data_type] = Avg_pearson_test
                    Results_pd.at[i,'p_value_test_' + data_type] = Avg_p_value_test

                # save results
                if Show_plots :
                    plot_loss_val(Avg_loss,Avg_val_loss,load_np_array=False)

                # if Save_plots :
                #     Full_path = (path + "/plot_vec/" + name + "_" + data_type + "/" + "Run_Idx_" + str(i)+ "/")
                #     os.mkdir(Full_path)
                #     np.save(Full_path + "loss",Avg_loss)
                #     np.save(Full_path + "val_loss",Avg_val_loss)

                if Debug:
                    # Print Average correlations:
                    print("A total number of "+str(AVG_TF_SEED_AMOUNT)+" Seeds were used to predict results")
                    print("Average pearson_correlation_val_" + data_type + " : " + str(Results_pd.at[i, 'pearson_correlation_val_'+ data_type]))
                    print("Average loss_mse_val_"+data_type+" : " + str(Results_pd.at[i, 'p_value_val_' + data_type]))

                    if Use_Test :
                        print("Average pearson_correlation_test_" + data_type + " : " + str(Results_pd.at[i, 'pearson_correlation_test_' + data_type]))
                        print("Average loss_mse_test_" + data_type + " : " + str(Results_pd.at[i, 'p_value_test_' + data_type]))

            if Save_Results :
                filename_csv = CSV_Path +"Run_idx_" + str(runidx) + ".csv"
                Results_pd.to_csv(filename_csv)
                print("Last Version Saved with " +str(i)+ " Interations")







if __name__ == "__main__":
    main()



