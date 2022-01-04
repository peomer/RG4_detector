import os

from tensorflow.keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Conv1D, MaxPooling1D, MaxPool1D
from keras.layers import Dropout
# from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, LSTM, GRU, Bidirectional, Input, \
    concatenate
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random as python_random
from tensorflow.keras.models import load_model
from csv import writer


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
                hotVec[i] = [0.25, 0.25, 0.25, 0.25, 0]
        return np.delete(hotVec, 4, 1)

def trim_seq(array, how_much):
    halp_p = len(array[0][1]) / 2
    from_idx = round(halp_p - how_much / 2)
    to_idx = round(halp_p + how_much / 2)
    trim = array.apply([lambda x: x.str.slice(from_idx, to_idx)])
    return trim

def get_data(path, min_read=2000, add_RNAplfold=False, export_np_arr=False, load_np_arr=False, add_evolution=False,
             add_mfe=False,bartel=False):
    # train
    if load_np_arr:
        folder = "/np_data"
        if add_RNAplfold:
            folder = folder + "_pl"
        if add_mfe:
            folder = folder + "_mfe"
        if add_evolution:
            folder = folder + "_evo"
        if bartel:
            folder = folder + "_bar"

        # Train
        X_train = np.load(path + folder + "/X_train.npy")
        y_train = np.load(path + folder + "/y_train.npy")
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

        if bartel:
            X_bartel = np.load(path+folder+'/X_bartel.npy')
            Y_bartel = np.load(path+folder+'/Y_bartel.npy')

    else:
        if bartel:
            with open(path + "/seq/bartel_seq") as source:
                X_bartel = np.array(list(map(one_hot_enc, source)))
            Y_bartel = np.load(path +'/np_data_bar/labels.npy')
        with open(path + "/seq/train-seq") as source:
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
        with open(path + "/seq/val-seq") as source:
            X_val = np.array(list(map(one_hot_enc, source)))
        y_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['rsr']).to_numpy()
        w_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['c_read']).to_numpy() + \
                pd.read_csv(path + '/csv_data/val_data.csv', usecols=['t_read']).to_numpy()
        chr_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['chromosome']).to_numpy()
        pos_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['position']).to_numpy()
        pos_val = pos_val.astype(float)
        strand_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['strand']).to_numpy()
        if add_mfe:
            mfe_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['mfe']).to_numpy()
        # test
        with open(path + "/seq/test-seq") as source:
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
        # print(len(ids))
        # set test min read
        ids = np.argwhere(w_test > min_read)[:, 0]
        X_test = X_test[ids]
        y_test = y_test[ids]
        w_test = w_test[ids]
        if add_mfe:
            mfe_test = mfe_test[ids]
        # print(len(ids))
        # scale_labels
        y_train = np.log(y_train)
        y_test = np.log(y_test)
        y_val = np.log(y_val)
        if bartel:
            Y_bartel = np.log(Y_bartel)

        if add_evolution:
            ref_dic = {}
            values = {}
            for i in range(23):
                if i == 0:
                    chrom = 'chrX'
                else:
                    chrom = 'chr' + str(i)
                file_evo = chrom + '.phastCons100way.wigFix'
                head_file = chrom + '_fixed_nums'
                with open('./evolutionary_conservation/' + file_evo) as source:
                    values_temp = list(source)
                values_temp = np.array(values_temp)
                with open('./evolutionary_conservation/' + head_file) as source:
                    head_nums = np.array(list(source))
                head_nums = head_nums.astype(int)
                idx = 0
                values_temp1 = np.zeros(1000000000)
                ref_dic[chrom] = []
                idx = 0
                for f in range(len(head_nums)):
                    head_nums[f] = head_nums[f] - 1
                    headline = values_temp[head_nums[f]]
                    ref_dic[chrom].append(get_headline(headline))
                    if f == 0:
                        continue
                    a = ref_dic[chrom][-1] - ref_dic[chrom][-2] - head_nums[f] + head_nums[f - 1] + 1
                    start = head_nums[f - 1] + 1
                    end = head_nums[f]
                    values_temp1[idx:idx + end - start] = values_temp[start:end]
                    idx = idx + end - start + a
                start = head_nums[f] + 1
                end = len(values_temp)
                values_temp1[idx:idx + end - start] = values_temp[head_nums[f] + 1:]
                idx = idx + end - start
                values[chrom] = values_temp1[:idx].astype(float)

                np.save("./rg4_data/np_data_evo/" + chrom + ".npy", values[chrom])
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
                a = str(chr_train[i])[2:-2]
                b = str(pos_train[i] - 140)
                c = str(pos_train[i] + 110)
                d = str(strand_train[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/train_plfold/" + plfold_name) as source:
                    pl_train = np.array(list(source))
                pl_train = pl_train.astype(float)
                temp = X_train[i]
                temp = np.column_stack((temp, pl_train))
                X_new.append(temp)
            X_train = np.array(X_new)
            X_new = []
            for i in range(len(X_val)):
                a = str(chr_val[i])[2:-2]
                b = str(pos_val[i] - 140)
                c = str(pos_val[i] + 110)
                d = str(strand_val[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/val_plfold/" + plfold_name) as source:
                    pl_val = np.array(list(source))
                pl_val = pl_val.astype(float)
                temp = X_val[i]
                temp = np.column_stack((temp, pl_val))
                X_new.append(temp)
            X_val = np.array(X_new)
            X_new = []
            for i in range(len(X_test)):
                a = str(chr_test[i])[2:-2]
                b = str(pos_test[i] - 140)
                c = str(pos_test[i] + 110)
                d = str(strand_test[i])[2:-2]
                e = ')_lunp\_clean'
                plfold_name = a + '_' + b + '-' + c + '(' + d + e
                with open(path + "/plfold/test_plfold/" + plfold_name) as source:
                    pl_test = np.array(list(source))
                pl_test = pl_test.astype(float)
                temp = X_test[i]
                temp = np.column_stack((temp, pl_test))
                X_new.append(temp)
            X_test = np.array(X_new)

    if export_np_arr:
        folder = "/np_data"

        if add_RNAplfold:
            folder = folder + "_pl"

        if add_mfe:
            folder = folder + "_mfe"

        if add_evolution:
            folder = folder + "_evo"
        if bartel:
            folder = folder + "_bar"

        np.save(path + folder + "/X_train.npy", X_train)
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
        if bartel:
            np.save(path + folder + "/X_bartel.npy",X_bartel)
            np.save(path + folder + "/Y_bartel.npy",Y_bartel)

    if add_mfe:
        return [X_train, y_train, mfe_train], [X_test, y_test, mfe_test], [X_val, y_val,
                                                                                            mfe_val]
    elif bartel:
        X_train_test = np.concatenate((X_train,X_test))
        y_train_test = np.concatenate((y_train,y_test))
        return [X_train_test, y_train_test], [X_val, y_val], [X_bartel, Y_bartel]
    else:
        return [X_train, y_train], [X_test, y_test], [X_val, y_val]


def get_headline(headline):
    for j in range(len(headline)):
        if (headline[j] == '=' and headline[j - 5:j] == 'start'):
            h = headline.find(' ', j)
            ref_num = headline[j + 1:h]
            break
    return (int(ref_num))


def trim_mat(data, INPUT_SIZE, mfe=False):
    if mfe:
        [X_Data, Y_Data, mfe_Data] = data
    else:
        [X_Data, Y_Data] = data
    total_data_size = X_Data.shape[1]
    start = total_data_size // 2 - INPUT_SIZE // 2
    end = start + INPUT_SIZE
    X_Data = X_Data[:, start:end, :]
    if mfe:
        data = [X_Data, Y_Data, mfe_Data]
    else:
        data = [X_Data, Y_Data]
    return data


class HyperParams:

    def __init__(self):
        self.input_size_list = [60, 80, 100, 120]
        self.conv_size_list = [16, 32, 64, 128]
        self.dense_size_list = [16, 32, 64, 128]
        self.dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lr_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.activations_list = ['relu', 'sigmoid', 'relu']
        self.batch_size_list = [16, 32, 64, 128]
        self.epochs_list = [x for x in range(3, 20)]
        self.INPUT_SIZE = 120
        self.FILTER = 64
        self.KERNEL_SIZE = 16
        self.POOLING = 1
        self.POOL_SIZE = 4
        self.DENCE_1 = 56
        self.DENCE_2 = 52
        self.ACTIVATION_1 = 'relu'
        self.ACTIVATION_2 = 'relu'
        self.DROPOUT_1 = 0.0
        self.DROPOUT_2 = 0.4
        self.TF_SEED = 1210
        self.EPOCH = 10
        self.BATCH_SIZE = 24
        self.CONV_PADDING = "valid"
        self.path = "./hparams/hyper_params1.csv"
        self.path = "./hparams/noseeds_sorted.csv"

    def rand_params(self):
        self.INPUT_SIZE = np.random.choice(self.input_size_list)
        self.FILTER = np.random.randint(8, 12) * 8
        self.KERNEL_SIZE = np.random.randint(1, self.INPUT_SIZE / 16 + 1) * 4
        self.POOLING = 1
        self.POOL_SIZE = np.random.randint(1, self.INPUT_SIZE / 16 + 1) * 4
        self.DENCE_1 = np.random.randint(6, 16) * 4
        self.DENCE_2 = np.random.randint(4, 16) * 4
        self.ACTIVATION_1 = np.random.choice(self.activations_list)
        self.ACTIVATION_2 = 'relu'
        self.DROPOUT_1 = np.random.randint(0, 7) * 0.1
        self.DROPOUT_2 = np.random.randint(0, 7) * 0.1
        self.TF_SEED = np.random.randint(10000)
        self.EPOCH = np.random.randint(5, 20)
        self.BATCH_SIZE = np.random.randint(1, 10) * 8

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

    def load_params(self, max=True, idx=0):
        Pearson_cor = pd.read_csv(self.path, usecols=['pearson_correlation_val_seq']).to_numpy()
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
            read_idx = max_idx
        else:
            read_idx = idx
        self.INPUT_SIZE = int(INPUT_SIZE[read_idx][0])
        self.FILTER = int(FILTER[read_idx][0])
        self.KERNEL_SIZE = int(KERNEL_SIZE[read_idx][0])
        self.POOL_SIZE = int(POOL_SIZE[read_idx][0])
        self.DENCE_1 = int(DENCE_1[read_idx][0])
        self.DENCE_2 = int(DENCE_2[read_idx][0])
        self.ACTIVATION_1 = str((ACTIVATION_1[read_idx])[0])
        self.ACTIVATION_2 = str((ACTIVATION_2[read_idx])[0])
        self.DROPOUT_1 = float(DROPOUT_1[read_idx][0])
        self.DROPOUT_2 = float(DROPOUT_2[read_idx][0])
        # self.TF_SEED = TF_SEED.astype(float)[read_idx]
        self.EPOCH = int(EPOCH[read_idx][0])
        self.BATCH_SIZE = int(BATCH_SIZE[read_idx][0])

        # self.INPUT_SIZE = INPUT_SIZE[read_idx].item()
        # self.FILTER = FILTER[read_idx].item()
        # self.KERNEL_SIZE = KERNEL_SIZE[read_idx].item()
        # self.POOL_SIZE = POOL_SIZE[read_idx].item()
        # self.DENCE_1 = DENCE_1[read_idx].item()
        # self.DENCE_2 = DENCE_2[read_idx].item()
        # self.ACTIVATION_1 = str((ACTIVATION_1[read_idx])[0])
        # self.ACTIVATION_2 = str((ACTIVATION_2[read_idx])[0])
        # self.DROPOUT_1 = DROPOUT_1[read_idx].item()
        # self.DROPOUT_2 = DROPOUT_2[read_idx].item()
        # self.TF_SEED = TF_SEED[read_idx].item()
        # self.EPOCH = EPOCH[read_idx].item()
        # self.BATCH_SIZE = BATCH_SIZE[read_idx].item()


        # self.INPUT_SIZE = np.asscalar(INPUT_SIZE[read_idx])
        # self.FILTER = np.asscalar(FILTER[read_idx])
        # self.KERNEL_SIZE = np.asscalar(KERNEL_SIZE[read_idx])
        # self.POOL_SIZE = np.asscalar(POOL_SIZE[read_idx])
        # self.DENCE_1 = np.asscalar(DENCE_1[read_idx])
        # self.DENCE_2 = np.asscalar(DENCE_2[read_idx])
        # self.ACTIVATION_1 = str((ACTIVATION_1[read_idx])[0])
        # self.ACTIVATION_2 = str((ACTIVATION_2[read_idx])[0])
        # self.DROPOUT_1 = np.asscalar(DROPOUT_1[read_idx])
        # self.DROPOUT_2 = np.asscalar(DROPOUT_2[read_idx])
        # self.TF_SEED = np.asscalar(TF_SEED[read_idx])
        # self.EPOCH = np.asscalar(EPOCH[read_idx])
        # self.BATCH_SIZE = np.asscalar(BATCH_SIZE[read_idx])

    def save_params(self):
        to_add = [self.INPUT_SIZE, self.FILTER, self.KERNEL_SIZE, self.POOL_SIZE, self.DENCE_1, self.DENCE_2,
                  self.ACTIVATION_1, self.ACTIVATION_2, self.DROPOUT_1, self.DROPOUT_2, self.TF_SEED, self.EPOCH,
                  self.BATCH_SIZE]
        with open(self.path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(to_add)
            f_object.close()


def build_mfe_model(hparams=HyperParams(), input_shape=None, mfe_shape=None):
    dataIn = Input(shape=input_shape)
    model1 = Conv1D(filters=hparams.FILTER, kernel_size=hparams.KERNEL_SIZE, input_shape=input_shape, name="conv",
                    padding=hparams.CONV_PADDING)(dataIn)
    model1 = MaxPool1D(pool_size=hparams.POOL_SIZE, name="pooling")(model1)
    model1 = Dropout(hparams.DROPOUT_1)(model1)
    model1 = Flatten()(model1)
    mfeIn = Input(shape=mfe_shape)
    model2 = Flatten()(mfeIn)
    model = concatenate([model1, model2])
    model = Dense(hparams.DENCE_1, activation=hparams.ACTIVATION_1, name="dense")(model)
    model = Dropout(hparams.DROPOUT_2)(model)
    model = Dense(hparams.DENCE_2, activation=hparams.ACTIVATION_2, name="dense2")(model)
    out = Dense(1, activation='linear', name="1dense")(model)
    model = Model([dataIn, mfeIn], out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    if Debug:
        model.summary()
    return model


def build_seq_model(Load_model=False, hparams=HyperParams(), input_shape=()):
    if not Load_model:
        model = Sequential()
        model.add(Conv1D(filters=hparams.FILTER, kernel_size=hparams.KERNEL_SIZE, input_shape=input_shape,
                         name="conv", padding=hparams.CONV_PADDING))
        model.add(MaxPool1D(pool_size=hparams.POOL_SIZE, name="pooling"))
        model.add(Dropout(hparams.DROPOUT_1))
        model.add(Flatten())
        model.add(Dense(hparams.DENCE_1, activation=hparams.ACTIVATION_1, name="dense"))
        model.add(Dense(hparams.DENCE_2, activation=hparams.ACTIVATION_2, name="dense2"))
        model.add(Dense(1, activation='linear', name="1dense"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        if Debug:
            model.summary()
    else:
        model = load_model('my_model')
    return model


def update_results_params(i=0, hparams=HyperParams()):
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


def eval_results(predictions=None, Y=None):
# def predict_results(model=Sequential(), X=None, Y=None):
    # predictions = model.predict(X, batch_size=len(X))  # Batch_Size defualt is 32
    # predictions = predictions.reshape(len(predictions))  # Reshape pred_val
    Y = Y.reshape(len(Y))  # Reshape Y_VAL
    # predictions = predictions.reshape(len(predictions))  # Reshape pred
    [pearson, p_value] = pearsonr(Y, predictions)
    mse = np.mean(np.square(Y - predictions))
    return [pearson[0], p_value, mse]



def plot_loss_val(loss=None, val_loss=None, load_np_array=False):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# ---------------------------------------------- End of functions def----------------------------------------------------


# ---------------------------------------------- Configure Params -------------------------------------------------------

import params

global Debug
Cloud_run = params.Cloud_run
Debug = params.Debug
save_model = params.save_model
Load_model = params.save_model
ITER = params.ITER
AVG_TF_SEED_AMOUNT = params.AVG_TF_SEED_AMOUNT
Show_plots = params.Show_plots
Save_plots = params.Save_plots
Data_run_list = params.Data_run_list
Save_Results = params.Save_Results
Use_Test = params.Use_Test

path = "./rg4_data"

if Debug:
    name = "debug"
    runidx = 0
    iterations = 1
    VERBOSE = 1
else:
    VERBOSE = 0
    name = "mega_run"
    runidx = input()
    iterations = ITER
# ---------------------------------------------- Configure DataFrame ----------------------------------------------------
# Results Data frame :
# adding Hyper Parapaters
Results_pd = pd.DataFrame()
# Result columns depending on the Data types to be Run.
witch_data_types = ''
for data_type in Data_run_list:
    witch_data_types = witch_data_types + data_type + "_"

CSV_Path = "./out_results/" + "out_results_csv_" + name + "_" + witch_data_types + "/"
os.makedirs(CSV_Path, exist_ok=True)
# ---------------------------------------------- Read data---------------------------------------------------------------
if 'seq' in Data_run_list:
    train, test, validation = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,
                                       add_evolution=False, add_mfe=False, bartel=False)
if 'bar' in Data_run_list:
    train_test, validation ,bartel_all = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,
                                       add_evolution=False, add_mfe=False, bartel=True)
if 'mfe' in Data_run_list:
    train_mfe, test_mfe, validation_mfe = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,
                                                   add_evolution=False, add_mfe=True, bartel=False)

if 'evo' in Data_run_list:
    train_evo, test_evo, validation_evo = get_data(path, load_np_arr=True, add_RNAplfold=False, export_np_arr=False,
                                                   add_evolution=True, add_mfe=False, bartel=False)

if 'plfold' in Data_run_list:
    train_pl, test_pl, validation_pl = get_data(path, load_np_arr=True, add_RNAplfold=True, export_np_arr=True,
                                                add_evolution=False, add_mfe=False, bartel=False)


# ---------------------------------------------- Main -------------------------------------------------------------------

def main(param_scan=True, Seed_scan=False):
    for i in range(ITER):  # Main Loop Runs ITER times
        hparams = HyperParams()  # Init Hyper paramaters

        if param_scan:
            hparams.rand_params()

        if Seed_scan:
            hparams.load_params(max=False, idx=i+1)  # Load parmas from CSV
            hparams.TF_SEED = np.random.randint(10000)  # Randomize Seed

        if Debug:
            hparams.print()

        TF_SEED_LIST = []
        for k in range(AVG_TF_SEED_AMOUNT):
            # Generate a random seed, Add Seed to List :
            TF_SEED_LIST.append(np.random.randint(10000))
        TF_SEED_LIST_STR = ''
        for s in TF_SEED_LIST:
            TF_SEED_LIST_STR += (str(s)+'_')
        hparams.TF_SEED = TF_SEED_LIST_STR
        # add them to Pandas
        update_results_params(i, hparams)

        # Trim Data
        if 'seq' in Data_run_list:
            [X_train, Y_train] = trim_mat(train, hparams.INPUT_SIZE)
            [X_test, Y_test] = trim_mat(test, hparams.INPUT_SIZE)
            [X_validation, Y_validation] = trim_mat(validation, hparams.INPUT_SIZE)

        if 'mfe' in Data_run_list:
            [X_train_mfe, Y_train_mfe, mfe_train] = trim_mat(train_mfe, hparams.INPUT_SIZE)
            [X_test_mfe, Y_test_mfe, mfe_test] = trim_mat(test_mfe, hparams.INPUT_SIZE)
            [X_validation_mfe, Y_validation_mfe, mfe_val] = trim_mat(validation_mfe,
                                                                                       hparams.INPUT_SIZE)

        if 'evo' in Data_run_list:
            [X_train_evo, Y_train_evo] = trim_mat(train_evo, hparams.INPUT_SIZE)
            [X_test_evo, Y_test_evo] = trim_mat(test_evo, hparams.INPUT_SIZE)
            [X_validation_evo, Y_validation_evo] = trim_mat(validation_evo, hparams.INPUT_SIZE)

        if 'plfold' in Data_run_list:
            [X_train_pl, Y_train_pl] = trim_mat(train_pl, hparams.INPUT_SIZE)
            [X_test_pl, Y_test_pl] = trim_mat(test_pl, hparams.INPUT_SIZE)
            [X_validation_pl, Y_validation_pl] = trim_mat(validation_pl, hparams.INPUT_SIZE)

        if 'bar' in Data_run_list:
            [X_train_test, Y_train_test] = trim_mat(train_test, hparams.INPUT_SIZE)
            [X_val, Y_val] = trim_mat(validation, hparams.INPUT_SIZE)
            [X_bartel, Y_bartel] = trim_mat(bartel_all, hparams.INPUT_SIZE)


        for data_type in Data_run_list:
            # prepare the data to be trained on:
            if data_type == 'mfe':
                X_TRAIN = [X_train_mfe, mfe_train]
                Y_TRAIN = Y_train_mfe
                X_VAL = [X_validation_mfe, mfe_val]
                Y_VAL = Y_validation_mfe
                X_TEST = [X_test_mfe, mfe_test]
                Y_TEST = Y_test_mfe
                mfe_shape = mfe_train.shape[1:]
            elif data_type == 'seq':
                X_TRAIN = X_train
                Y_TRAIN = Y_train
                X_VAL = X_validation
                Y_VAL = Y_validation
                X_TEST = X_test
                Y_TEST = Y_test
            elif data_type == 'evo':
                X_TRAIN = X_train_evo
                Y_TRAIN = Y_train_evo
                X_VAL = X_validation_evo
                Y_VAL = Y_validation_evo
                X_TEST = X_test_evo
                Y_TEST = Y_test_evo
            elif data_type == 'plfold':
                X_TRAIN = X_train_pl
                Y_TRAIN = Y_train_pl
                X_VAL = X_validation_pl
                Y_VAL = Y_validation_pl
                X_TEST = X_test_pl
                Y_TEST = Y_test_pl
            elif data_type == 'bar':
                X_TRAIN = X_train_test
                Y_TRAIN = Y_train_test
                X_VAL = X_val
                Y_VAL = Y_val
                X_TEST = X_bartel
                Y_TEST = Y_bartel

            data_shape = X_TRAIN.shape[1:]
            if data_shape[0]==None:
                print('none')
                continue


            # elif data_type == 'from_bartel':
            #     X_TRAIN = np.concatenate((X_train_bartel,X_test_bartel), axis=0)
            #     Y_TRAIN = np.concatenate((Y_train_bartel,Y_test_bartel), axis=0)
            #     X_VAL = X_val_bartel
            #     Y_VAL = Y_val_bartel
            #     X_TEST = np.concatenate((X_train,X_test,X_validation),axis=0)
            #     Y_TEST = np.concatenate((Y_train,Y_test,Y_validation),axis=0)
            #     data_shape = X_train.shape[1:]

            # Reset Lists for Averging over TF Seeds
            loss_list = []
            val_loss_list = []
            predictions_list_test = []
            predictions_list = []
            for tf_seed in TF_SEED_LIST:
                # hparams.TF_SEED = tf_seed

                # Set the TF Seed before the model build:

                # The below is necessary for starting Numpy generated random numbers
                # in a well-defined initial state.
                # np.random.seed(TF_SEED)

                # The below is necessary for starting core Python generated random numbers
                # in a well-defined state.
                # python_random.seed(TF_SEED)

                # The below set_seed() will make random number generation
                # in the TensorFlow backend have a well-defined initial state.
                # For further details, see:  https://www.tensorflow.org/api_docs/python/tf/random/set_seed
                tf.random.set_seed(tf_seed)

                if data_type == 'mfe':
                    model = build_mfe_model(hparams=hparams, mfe_shape=mfe_shape, input_shape=data_shape)
                else:
                    model = build_seq_model(hparams=hparams, Load_model=False, input_shape=data_shape)

                if Debug:
                    hparams.EPOCH = 10

                # Train Model

                model.fit(x=X_TRAIN, y=Y_TRAIN, epochs=hparams.EPOCH, verbose=VERBOSE, batch_size=hparams.BATCH_SIZE,
                          validation_data=(X_VAL, Y_VAL))

                # Plot Loss for val and training
                if Debug:
                    # summarize history for loss
                    loss_list.append(model.history.history['loss'])
                    val_loss_list.append(model.history.history['val_loss'])

                # Predictions for Test
                if Use_Test:
                    predictions = model.predict(X_TEST, batch_size=len(X_TEST))  # Batch_Size defualt is 32
                    predictions_list_test.append(predictions)

                # Predictions for Validation
                predictions = model.predict(X_VAL, batch_size=len(X_VAL))  # Batch_Size defualt is 32
                predictions_list.append(predictions)

            # Average from all AVG_TF_SEED_PER Validation
            Avg_loss = np.average(loss_list, axis=0)  # Average done on Axis 0 meaning on all first elements etc
            Avg_val_loss = np.average(val_loss_list, axis=0)
            predictions = np.average(predictions_list, axis=0)
            [pearson_cor, p_value, mse] = eval_results(predictions,Y_VAL)

            # Update results Data frame Validation :
            Results_pd.at[i, 'pearson_correlation_val_' + data_type] = pearson_cor
            Results_pd.at[i, 'p_value_val_' + data_type] = p_value
            Results_pd.at[i, 'loss_mse_val_' + data_type] = mse

            if Use_Test:
                # Average from all AVG_TF_SEED_PER Test:
                # Update results Data frame Test :
                predictions_test = np.average(predictions_list_test, axis=0)
                [pearson_cor_test, p_value_test, mse_test] = eval_results(predictions_test,Y_TEST)
                Results_pd.at[i, 'loss_mse_test_' + data_type] = mse_test
                Results_pd.at[i, 'pearson_correlation_test_' + data_type] = pearson_cor_test
                Results_pd.at[i, 'p_value_test_' + data_type] = p_value_test

            # save results
            if Show_plots:
                plot_loss_val(Avg_loss, Avg_val_loss, load_np_array=False)

            # if Save_plots :
            #     Full_path = (path + "/plot_vec/" + name + "_" + data_type + "/" + "Run_Idx_" + str(i)+ "/")
            #     os.mkdir(Full_path)
            #     np.save(Full_path + "loss",Avg_loss)
            #     np.save(Full_path + "val_loss",Avg_val_loss)

            if Debug:
                # Print Average correlations:
                print("A total number of " + str(AVG_TF_SEED_AMOUNT) + " Seeds were used to predict results")
                print("Average pearson_correlation_val_" + data_type + " : " + str(
                    Results_pd.at[i, 'pearson_correlation_val_' + data_type]))
                print("Average loss_mse_val_" + data_type + " : " + str(Results_pd.at[i, 'p_value_val_' + data_type]))

                if Use_Test:
                    print("Average pearson_correlation_test_" + data_type + " : " + str(
                        Results_pd.at[i, 'pearson_correlation_test_' + data_type]))
                    print("Average loss_mse_test_" + data_type + " : " + str(
                        Results_pd.at[i, 'p_value_test_' + data_type]))

        if (Save_Results ):
        # if (Save_Results and i%10 ==0):
            filename_csv = CSV_Path + "Run_idx_" + str(runidx) + ".csv"
            Results_pd.to_csv(filename_csv)
            print("Last Version Saved with " + str(i) + " Interations")


if __name__ == "__main__":
    # main(Seed_scan=False, param_scan=True)
    print("Done with param scan")
    main(Seed_scan=True,param_scan=False)
