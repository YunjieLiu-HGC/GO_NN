import datetime
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath
import argparse, math, random, sys, time
import numpy as np
#np.random.seed(0)  # set random seed for keras neural nets
#import xgboost as xgb
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

#input_files=["../data/Chi/GONET_tpm_dataset_1p.csv","../data/Chi/GONET_tpm_dataset_5p.csv","../data/Chi/GONET_tpm_dataset_10p.csv","../data/Liver/GONET_tpm_dataset_10p.csv","../data/Liver/GONET_tpm_dataset_10p.csv","../data/Liver/GONET_tpm_dataset_10p.csv"]
#input_files=["../data/Chi/GONET_tpm_dataset_1p.csv","../data/Chi/GONET_tpm_dataset_5p.csv","../data/Chi/GONET_tpm_dataset_10p.csv"]
input_files=["../data/folds/fold_dia_1p/","../data/folds/fold_dia_5p/","../data/folds/fold_dia_10p/","../data/folds/fold_lc_1p/","../data/folds/fold_lc_5p/","../data/folds/fold_lc_10p/"]
input_labels=["Chi_1p","Chi_5p","Chi_10p","lc_1p","lc_5p","lc_10p"]
#input_labels=["Chi_1p","Chi_5p","Chi_10p","LC_1p","LC_5p","LC_10p"]
y_files=["../data/Chi/GONET_dia_status.csv","../data/Chi/GONET_dia_status.csv","../data/Chi/GONET_dia_status.csv"]
#y_files=["../data/Chi/GONET_dia_status.csv","../data/Chi/GONET_dia_status.csv","../data/Chi/GONET_dia_status.csv","../data/Liver/GONET_lc_status.csv","../data/Liver/GONET_lc_status.csv","../data/Liver/GONET_lc_status.csv"]
def run_rf(train_X, test_X, train_y, test_y, depth=6, est=100, c='gini',seed=0):
    clf = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=c, random_state=seed)
    clf.fit(train_X, train_y)
    ypred = np.array([i[1] for i in clf.predict_proba(test_X)])
    metrics = gen_eval_metrics(test_y, ypred)
    accuracy = metrics[0]
    #cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
    #accuracy = cor / len(test_y)
    print('Fold accuracy: ' + str(accuracy))
    return metrics
def run_svm(train_X, test_X, train_y, test_y, c=1.0, kern='linear', seed=0):
    clf = SVC(C=c, kernel=kern, random_state=seed, probability=True)
    clf.fit(train_X, train_y)
    ypred = np.array([i[1] for i in clf.predict_proba(test_X)])
    ypred_binary = clf.predict(test_X)
    metrics = gen_eval_metrics(test_y, ypred, svm_binary=ypred_binary)
    accuracy = metrics[0]
    #cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
    #accuracy = cor / len(test_y)
    print('Fold accuracy: ' + str(accuracy))
def gen_eval_metrics(y_true, y_pred, svm_binary=[]):
    auc = roc_auc_score(y_true, y_pred)
    if len(svm_binary) > 0:  # a workaround for inaccurate SVC predict_proba
        y_pred = svm_binary
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] >= 0.5:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if y_pred[i] < 0.5:
                tn += 1.0
            else:
                fp += 1.0
    if tp > 0.0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    auprc=average_precision_score(y_true, y_pred)
    return [accuracy, precision, recall, f1, auc, auprc]

def build_and_fit_model(train_X, test_X, train_y, test_y,
                        numlayers=5, dropout=0.25, opt='adam',learn_rate=0.001):
    model = Sequential()
    # define initial layer size and uniform scaling-down factor per layer
    layersize, layer_scale = len(train_X[0]), 1.0 / float(numlayers + 1)

    # input layer, then scaled down fully connected layers, then output layer
    model.add(Dense(layersize, input_dim=layersize,
        kernel_initializer='normal', activation='relu'))
    for i in range(numlayers):
        this_layersize = layersize - int(layersize * (layer_scale * (i+1)))
        model.add(Dense(this_layersize,
            kernel_initializer='normal', activation='tanh'))
        model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    #opt = optimizers.adam(lr=0.0001)
    if opt == 'adam':
        opt = optimizers.Adam(lr=learn_rate/10.0)  # adam needs slower rate
    elif opt == 'sgd':
        opt = optimizers.SGD(lr=learn_rate)
    else:
        opt = optimizers.Adagrad(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(train_X, train_y, epochs = 50, verbose=0)
    print('# of trainable params of the model is %s' % model.count_params())
    ypred = np.array([i[0] for i in model.predict(test_X, batch_size=32)])
    metrics = gen_eval_metrics(test_y, ypred)
    accuracy = metrics[0]
    print('Fold accuracy: ' + str(accuracy))
    #score = model.evaluate(test_X, test_y, batch_size=32)
    return metrics

res_nn = []
for fi in range(len(input_files)):
    print(fi)
    #X = pd.read_csv(input_files[fi], header=0, low_memory=False)
    #Y = pd.read_csv(y_files[fi], header=0)
    #pool = set(np.arange(len(X)))#set of all samples
    num_folds, fold_accs = 5, []
    #arrayx = X.iloc[:,1:].to_numpy()
    #arrayy = Y.to_numpy()
    #arrayx[np.isnan(arrayx)] = 0
    #arrayy =arrayy.astype(int)
    numlayers_arr = [1,2,3]
    dropout_arr = [0.0]
    opt_arr = ['adam']
    lr_arr = [0.001]

    for numlayers in numlayers_arr:
        for drop in dropout_arr:
            for opt in opt_arr:
                for lr in lr_arr:
                    fold_metrics=[]
                    for i in range(num_folds):
                        print("Running Cross Validation %d" %(i+1))
                        #fold_size = int(len(X) / num_folds) + ((i+1) > (num_folds - (len(X) % num_folds)))
                        #test_indices = list(random.sample(pool, fold_size))
                        #trainset = pool.difference(test_indices)  # remove this fold from set
                        # Training set is patients not in test set
                        #train_indices = list(trainset)
                        # Now grab the actual feature vectors for train and test sets
                        #train_X, test_X = arrayx[train_indices][:], arrayx[test_indices][:]
                        #train_y, test_y = arrayy[train_indices], arrayy[test_indices]
                        train_X=pd.read_csv(input_files[fi]+'train_x'+str(i)+'.csv',header=None)
                        train_y=pd.read_csv(input_files[fi]+'train_y'+str(i)+'.csv',header=None)
                        test_X=pd.read_csv(input_files[fi]+'test_x'+str(i)+'.csv',header=None)
                        test_y=pd.read_csv(input_files[fi]+'test_y'+str(i)+'.csv',header=None)
                        train_X=train_X.to_numpy()
                        train_y=train_y.to_numpy()
                        test_X=test_X.to_numpy()
                        test_y=test_y.to_numpy()
                        metrics = build_and_fit_model(train_X, test_X, train_y, test_y,numlayers=numlayers,dropout=drop,opt=opt,learn_rate=lr)
                        fold_metrics.append(metrics)
                    acc,pre,rec,f1,auc,auprc = [],[],[],[],[],[]
                    for met in fold_metrics:
                        acc.append(met[0])
                        pre.append(met[1])
                        rec.append(met[2])
                        f1.append(met[3])
                        auc.append(met[4])
                        auprc.append(met[5])
                    res_nn.append({'label':input_labels[fi],'numlayers':numlayers,'dropout_rate':drop,'optimizer':opt,'learn_rate':lr,'Accuracy':np.average(acc),'Accuracy_sd':math.sqrt(np.var(acc)),'Precision':np.average(pre),'Precision_sd':math.sqrt(np.var(pre)),'Recall':np.average(rec),'Recall_sd':math.sqrt(np.var(rec)),'F1Score':np.average(f1),'F1Score_sd':math.sqrt(np.var(f1)),'AUC':np.average(auc),'AUC_sd':math.sqrt(np.var(auc)),'AUPRC':np.average(auprc),'AUPRC_sd':math.sqrt(np.var(auprc))})

res_nndf = pd.DataFrame(res_nn)
res_nndf.to_csv("metrics.csv",index=False)

