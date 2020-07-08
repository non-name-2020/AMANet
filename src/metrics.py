#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,average_precision_score,accuracy_score,precision_score,recall_score,f1_score


def roc_auc_multi(target, prob):
    return roc_auc_score(target, prob, average='macro')


def prc_auc_multi(target, prob):
    return average_precision_score(target, prob, average='macro')


def roc_auc_multi1(y_gt, y_prob):
    all_micro = []
    print(y_gt)
    print(y_prob)
    for b in range(len(y_gt)):
        all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
    return np.mean(all_micro)

def precision_auc(y_gt, y_prob):
    all_micro = []
    for b in range(len(y_gt)):
        all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
    return np.mean(all_micro)

def metrics_multi(target, predict):
    target_len = len(target)
    predict_len = len(predict)
    union_len = len(set(target).union(set(predict)))
    intersection_len = len(set(target).intersection(set(predict)))
    prec = 0.0
    if predict_len > 0:
        prec = intersection_len / predict_len
    recall = intersection_len / target_len
    jac = intersection_len / union_len
    if prec + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * recall / (prec + recall)
    return prec, recall, jac, f1


def roc_auc_non_multi(target, predict):
    return roc_auc_score(target, predict, average='macro')

def prc_auc_non_multi(target, prob):
    return average_precision_score(target, prob, average='macro')

def metrics_non_multi(target, predict):
    c = confusion_matrix(target, predict)
    tp = c[0][0]
    fn = c[0][1]
    fp = c[1][0]
    tn = c[1][1]
    if tp + fp <= 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if tp + fn <= 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    acc = (tp + tn) / (tp + fn + fp + tn)
    if prec + recall <= 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return acc, prec, recall, f1


def metrics_multi_class(y_true, y_pred):
    if not isinstance(y_true, list):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', labels=y_true)
    recall = recall_score(y_true, y_pred, average='macro', labels=y_true)
    f1 = f1_score(y_true, y_pred, average='macro', labels=y_true)
    return acc, prec, recall, f1


def roc_auc_multi_class(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='macro')


def prc_auc_multi_class(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='macro')

'''
y_true = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,0.0,0.0]])
y_pred = np.array([[0.9,0.1,0],[0.2,0.8,0.0],[0.9,0.1,0],[0.9,0.1,0],[0.2,0.8,0.0],[0.7,0.1,0.2]])
print(np.unique(np.argmax(y_pred, axis=1)))
print(metrics_multi_class(y_true, y_pred))
#(0.8333333333333334, 0.5833333333333334, 0.6666666666666666, 0.6190476190476191)
print(roc_auc_multi_class(y_true, y_pred))

print(prc_auc_multi_class(y_true, y_pred))
'''