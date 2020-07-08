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

from dataset import load_tax_data
from metrics import metrics_non_multi, roc_auc_non_multi,prc_auc_non_multi

np.random.seed(1203)
model_name = 'LR-tax'
model_save_name = ''

# time
time_str = time.strftime("%Y%m%d%H%M", time.localtime())


def create_dataset(df, voc_size):
    i1_len = voc_size[0]
    i2_len = voc_size[1]
    output_len = voc_size[2]
    input_len = i1_len + i2_len
    X = []
    y = []
    for index in range(df.shape[0]):
        sample = df[index:index+1]
        # input seq1
        input_seq1 = eval(sample["sb_val_qcut"].values[0])
        # inout seq2
        input_seq2 = eval(sample["fp_val_qcut"].values[0])
        # output seq
        output = sample["type"].values[0]
        if output == 'xk':
            o = 0
        else:
            o = 1

        multi_hot_input = np.zeros(input_len)
        multi_hot_input[input_seq1] = 1
        multi_hot_input[np.array(input_seq2) + i1_len] = 1
        X.append(multi_hot_input)
        y.append(o)

    return np.array(X), np.array(y)


def main(datapath='../../../tax-data/records.csv'):
    # model save dir
    model_save_dir = os.path.join("../../../model/tax-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # train log save dir
    log_save_dir = os.path.join("../../../logs/tax-task", model_name, time_str)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    # eval logs
    file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        
    data_train, data_valid, data_test, voc_size = load_tax_data(data_path=datapath)
    train_X, train_y = create_dataset(data_train, voc_size)
    eval_X, eval_y = create_dataset(data_test, voc_size)

    start_time = time.time()

    model = LogisticRegression()
    model.fit(train_X, train_y)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    y_pred = model.predict(eval_X)
    y_prob = model.predict_proba(eval_X)
    y_prob = y_prob[:, 1]

    acc, prec, recall, f1 = metrics_non_multi(eval_y, y_pred)
    roc_auc = roc_auc_non_multi(eval_y, y_prob)
    prauc = prc_auc_non_multi(eval_y, y_prob)
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
    main(datapath='../../../tax-data/records.csv')