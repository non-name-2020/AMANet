#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
from utils import multi_label_metric
from dataset import load_data

data_train, data_valid, data_test, voc, voc_size = load_data(data_path='../../../medical-data/records_final.pkl',
                                                            voc_path='../../../medical-data/voc_final.pkl')
med_voc = voc['med_voc']

# nearest
def main():
    gt = []
    pred = []
    for patient in data_test:
        gt.append(patient[0][2])
        pred.append([])
        if len(patient) == 1:
            continue
        for adm_idx, adm in enumerate(patient):
            if adm_idx < len(patient) - 1:
                gt.append(patient[adm_idx+1][2])
                pred.append(adm[2])
    med_voc_size = len(med_voc.idx2word)
    y_gt = np.zeros((len(gt), med_voc_size))
    y_pred = np.zeros((len(gt), med_voc_size))
    for idx, item in enumerate(gt):
        #print(item)
        y_gt[idx, item] = 1
    for idx, item in enumerate(pred):
        y_pred[idx, item] = 1

    ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_pred)

    print('Jaccard: %.4f,  ROC_AUC: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (ja, roc_auc, prauc, avg_p, avg_r, avg_f1))

if __name__ == '__main__':
    main()