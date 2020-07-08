#!/usr/bin/env python
# encoding: utf-8

import dill
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import os
import sys
sys.path.append("..")
sys.path.append("../..")
from utils import multi_label_metric
from dataset import load_data


np.random.seed(1203)
model_name = 'LR'

if not os.path.exists(os.path.join("res", model_name)):
    os.makedirs(os.path.join("res", model_name))


def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)


def main():
    grid_search = False

    data_train, data_valid, data_test, voc, voc_size = load_data(data_path= '../../../medical-data/records_final.pkl',
                                                                voc_path='../../../medical-data/voc_final.pkl')
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)

    start_time = time.time()
    if grid_search:
        params = {
            'estimator__penalty': ['l2'],
            'estimator__C': np.linspace(0.00002, 1, 100)
        }

        model = LogisticRegression()
        classifier = OneVsRestClassifier(model)
        lr_gs = GridSearchCV(classifier, params, verbose=1).fit(train_X, train_y)

        print("Best Params", lr_gs.best_params_)
        print("Best Score", lr_gs.best_score_)

        return

    model = LogisticRegression(C=0.90909)
    classifier = OneVsRestClassifier(model)
    classifier.fit(train_X, train_y)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("\rspend time to train: %.2f min" % elapsed_time)


    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len

    jas = []
    roc_aucs = []
    praucs  = []
    avg_ps = []
    avg_rs = []
    avg_f1s = []
    for patient in data_test:
        X = []
        y = []
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

        X = np.array(X)
        y = np.array(y)
        y_pred = classifier.predict(X)
        y_prob = classifier.predict_proba(X)
        ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y, y_pred, y_prob)
        jas.append(ja)
        roc_aucs.append(roc_auc)
        praucs.append(prauc)
        avg_ps.append(avg_p)
        avg_rs.append(avg_r)
        avg_f1s.append(avg_f1)

    print('jaccard: %.4f, roc_auc: %.4f, prauc: %.4f, avg_prc: %.4f, avg_recall: %.4f, avg_f1: %.4f\n'
          % (np.mean(jas), np.mean(roc_aucs), np.mean(praucs), np.mean(avg_ps), np.mean(avg_rs), np.mean(avg_f1s)))

    history = defaultdict(list)
    history['jaccard'].append(np.mean(jas))
    history['avg_p'].append(np.mean(avg_ps))
    history['avg_r'].append(np.mean(avg_rs))
    history['avg_f1'].append(np.mean(avg_f1s))
    history['prauc'].append(np.mean(praucs))
    history['roc_auc'].append(np.mean(roc_aucs))

    dill.dump(history, open(os.path.join('res', model_name, 'train_history.pkl'), 'wb'))

if __name__ == '__main__':
    main()