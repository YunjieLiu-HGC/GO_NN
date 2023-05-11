import logging

import numpy as np
import pandas as pd

from config_path import *
from sklearn.model_selection import train_test_split
data_path = DATA_PATH
lc_path=LC_DATA_PATH



class LCData():

    def __init__(self,feature='gene_1p', data_type='mut', account_for_data_type=None, cnv_levels=5,
                 cnv_filter_single_event=True, mut_binary=False,
                 selected_genes=None, combine_type='intersection',
                 use_coding_genes_only=False, drop_AR=False,
                 balanced_data=False, cnv_split=False,
                 shuffle=False, selected_samples=None, training_split=0):

        self.training_split = training_split
        if feature=='gene_1p':
            tpm_file = join(lc_path,'GONET_tpm_dataset_1p.csv')
        elif feature=='gene_5p':
            tpm_file = join(lc_path,'GONET_tpm_dataset_5p.csv')
        elif feature=='gene_10p':
            tpm_file = join(lc_path,'GONET_tpm_dataset_10p.csv')

        #tpm_file = join(lc_path,'GONET_tpm_dataset_1p.csv')
        #tpm_file = join(lc_path,'GONET_tpm_dataset_5p.csv')
        #tpm_file = join(lc_path,'GONET_tpm_dataset_10p.csv')
        #tpm_file = join(lc_path,'GONET_lc_tpm.csv')
        dia_file = join(lc_path,'GONET_lc_status.csv')
        id_file = join(lc_path,'lc_with_id.csv')
        tpm = pd.read_csv(tpm_file, header=0, low_memory=False)
        dia = pd.read_csv(dia_file, header=0)
        iddf = pd.read_csv(id_file, header=0)
        x = tpm
        y = dia
        rows = iddf['id']
        cols = tpm.columns

        if type(x) == pd.DataFrame:
            x = x.values
        if type(y) == pd.DataFrame:
            y = y.values

        if balanced_data:
            pos_ind = np.where(y == 1.)[0]
            neg_ind = np.where(y == 0.)[0]

            n_pos = pos_ind.shape[0]
            n_neg = neg_ind.shape[0]
            n = min(n_pos, n_neg)

            pos_ind = np.random.choice(pos_ind, size=n, replace=False)
            neg_ind = np.random.choice(neg_ind, size=n, replace=False)

            ind = np.sort(np.concatenate([pos_ind, neg_ind]))

            y = y[ind]
            x = x[ind,]
            rows = rows[ind]

        if shuffle:
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            y = y[ind, :]
            rows = rows[ind]

        self.x = x
        self.y = y
        self.info = rows
        self.columns = cols

    def get_data(self):
        return self.x, self.y, self.info, self.columns

    def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns
        #splits_path = join(DIABETES_DATA_PATH, 'splits')

        #training_file = 'training_set_{}.csv'.format(self.training_split)
        #training_set = pd.read_csv(join(splits_path, training_file))

        #validation_set = pd.read_csv(join(splits_path, 'validation_set.csv'))
        #testing_set = pd.read_csv(join(splits_path, 'test_set.csv'))

        #info_train = list(set(info).intersection(training_set.id))
        #info_validate = list(set(info).intersection(validation_set.id))
        #info_test = list(set(info).intersection(testing_set.id))

        #ind_train = info.isin(info_train)
        #ind_validate = info.isin(info_validate)
        #ind_test = info.isin(info_test)

        #x_train = x[ind_train]
        #x_test = x[ind_test]
        #x_validate = x[ind_validate]

        #y_train = y[ind_train]
        #y_test = y[ind_test]
        #y_validate = y[ind_validate]

        #info_train = info[ind_train]
        #info_test = info[ind_test]
        #info_validate = info[ind_valida
        x_train, x_test, y_train, y_test, info_train, info_test = train_test_split(x,y,info,train_size=0.8,test_size=0.2, random_state=1234)
        x_test, x_validate, y_test, y_validate, info_test, info_validate = train_test_split(x_test,y_test,info_test,train_size=0.5,test_size=0.5, random_state=1234)
        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns

