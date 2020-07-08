#!/usr/bin/env python
# encoding: utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../..")
import dill
import time
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


from dataset import load_drgs_data, prepare_drgs_dual
from utils import llprint
from model_keras_new import build
from metrics import metrics_multi_class, roc_auc_multi_class, prc_auc_multi_class

np.random.seed(1203)
model_name = 'LR-drgs'
model_save_name = ''

# time
time_str = time.strftime("%Y%m%d%H%M", time.localtime())


def create_dataset(df, voc_size):
    i1_len = voc_size[0]
    i2_len = voc_size[1]
    drgs_size = voc_size[2]
    input_len = i1_len + i2_len
    print("input_len:%d" %input_len)
    X = []
    y = []
    y_1 = []
    for index in range(df.shape[0]):
        sample = df[index:index+1]
        # input seq1
        input_seq1 = eval(sample["diagnosis"].values[0])
        # inout seq2
        input_seq2 = eval(sample["operation"].values[0])
        # output seq
        output = sample["drgs"].values[0]

        o = [0] * drgs_size
        o[output] = 1

        multi_hot_input = np.zeros(input_len)
        multi_hot_input[input_seq1] = 1
        multi_hot_input[np.array(input_seq2) + i1_len] = 1
        X.append(multi_hot_input)
        y.append(output)
        y_1.append(o)
    return np.array(X), np.array(y), np.array(y_1)


def main(datapath='../../../drgs-data/records.csv'):
    # model save dir
    model_save_dir = os.path.join("../../../model/drgs-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # train log save dir
    log_save_dir = os.path.join("../../../logs/drgs-task", model_name, time_str)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    # eval logs
    file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        
    data_train, data_valid, data_test, voc_size = load_drgs_data(data_path=datapath)
    train_X, train_y, train_y_1 = create_dataset(data_train, voc_size)
    eval_X, eval_y, eval_y_1 = create_dataset(data_test, voc_size)

    start_time = time.time()

    model = LogisticRegression()
    model.fit(train_X, train_y)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    y_pred = model.predict(eval_X)
    y_prob = model.predict_proba(eval_X)

    acc, prec, recall, f1 = metrics_multi_class(eval_y_1, y_prob)
    roc_auc = roc_auc_multi_class(eval_y_1, y_prob)
    prauc = prc_auc_multi_class(eval_y_1, y_prob)
    print("spend time to train: %.2f min" % elapsed_time)
    print('acc: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n' % (acc, prec, recall, f1, prauc, roc_auc))
    file.write("spend time to train: %.2f min\n" % elapsed_time)
    file.write("test acc: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n" % (acc, prec, recall, f1, prauc, roc_auc))
    file.write("###############################################################\n")

    pickle.dump(model, open(os.path.join(model_save_dir, 'model_%s_%.4f.h5' % (time_str, f1)), 'wb'))
    history = defaultdict(list)
    history['acc'].append(acc)
    history['prec'].append(prec)
    history['recall'].append(recall)
    history['f1'].append(f1)
    history['prauc'].append(prauc)
    history['roc_auc'].append(roc_auc)

    dill.dump(history, open(os.path.join(model_save_dir, 'train_history_%s.pkl' % time_str), 'wb'))
    file.close()

if __name__ == '__main__':
    main(datapath='../../../drgs-data/records.csv')