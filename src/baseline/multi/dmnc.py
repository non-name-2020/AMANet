#!/usr/bin/env python
# encoding: utf-8

import os
import dill
import time
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from collections import defaultdict
from sklearn.utils import shuffle


from models import DMNC
from utils import llprint, sequence_metric, get_n_params
from dataset import load_data

torch.manual_seed(1203)
model_name = 'DMNC-medical'
model_save_name = ''
time_str = time.strftime("%Y%m%d%H%M", time.localtime())

EPOCH = 15
LR = 0.0005
TEST = False

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]
    out_list = []
    for i in range(len(pind)):
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                continue
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict

# evaluate
def model_eval(model, data_test, voc_size):
    print("#####################%eval#####################")
    model.eval()
    ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    records = []
    eval_size = len(data_test)
    for step, input in enumerate(data_test):
        llprint("\rBatch: %d/%d" % (step + 1, eval_size))
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        i1_state, i2_state, i3_state = None, None, None
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted_predict)
            y_pred_prob.append(np.mean(output_logits[:,:-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
        records.append(y_pred_label)

        adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred),
                                                                              np.array(y_pred_prob),
                                                                              np.array(y_pred_label))
        ja.append(adm_ja)
        roc_auc.append(adm_roc_auc)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    print('')
    return np.mean(ja), np.mean(roc_auc), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    # model save dir
    model_save_dir = os.path.join("../../../model/medical-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    data_train, data_valid, data_test, voc, voc_size = load_data(data_path='../../../medical-data/records_final.pkl',
                                                                voc_path='../../../medical-data/voc_final.pkl')
    train_size = len(data_train)

    END_TOKEN = voc_size[2] + 1

    model = DMNC(voc_size)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join(model_save_dir, model_save_name), 'rb')))
        adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = model_eval(model, data_test, voc_size)
        print('Jaccard: %.4f,  ROC_AUC: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n'
                % (adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1))
    else:
        # train log save dir
        log_save_dir = os.path.join("../../../logs/medical-task", model_name, time_str)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

        # eval logs
        file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        file.write("Number of parameters: %d\n" % get_n_params(model))

        criterion2 = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LR)
        history = defaultdict(list)
        best_jac = 0.0
        best_epoch = 1
        best_model = None

        for epoch in range(EPOCH):
            file.write("Epoch: %d/%d\n" % (epoch + 1, EPOCH))
            llprint('Epoch: %d/%d\n' % (epoch + 1, EPOCH))
            loss_record = []
            start_time = time.time()
            data_train = shuffle(data_train)
            model.train()
            for step, input in enumerate(data_train):
                llprint("\rBatch %d/%d" % (step + 1, train_size))
                i1_state, i2_state, i3_state = None, None, None
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
                    loss = criterion2(output_logits, torch.LongTensor(loss_target))

                    loss_record.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
            print('')
            ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = model_eval(model, data_test, voc_size)
            history['ja'].append(ja)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['roc_auc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60

            file.write("spend time to train: %.2f min\n" % elapsed_time)
            file.write("train loss: %.6f\n" % (np.mean(loss_record)))
            file.write("test jac: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n" % (ja, avg_p, avg_r, avg_f1, prauc, roc_auc))
            file.write("###############################################################\n")

            print("spend time to train: %.2f min" % elapsed_time)
            print("train loss: %.6f" % (np.mean(loss_record)))
            print("test jad: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f" % (ja, avg_p, avg_r, avg_f1, prauc, roc_auc))
            print("###############################################################\n")

            print('Epoch: %d, loss: %.4f, One epoch time: %.2fm, Appro left time: %.2fh\n' % (epoch + 1,
                                                                                                np.mean(loss_record),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                        EPOCH - epoch - 1)/60))

            model_save_path = os.path.join(model_save_dir, 'model_%d_%s_%.4f.h5' % ((epoch + 1), time_str, ja))
            torch.save(model.state_dict(), open(model_save_path, 'wb'))
            if best_jac < ja:
                best_jac = ja
                best_epoch = epoch + 1
                best_model = model_save_path
            file.flush()

        dill.dump(history, open(os.path.join(model_save_dir, 'train_history_%s.pkl' % time_str), 'wb'))
        os.rename(best_model, best_model.replace(".h5", "_best.h5"))
        print("train done. best epoch: %d, best: jac: %f, model path: %s" % (best_epoch, best_jac, best_model))
        file.write("train done. best epoch: %d, best: jac: %f, model path: %s\n" % (best_epoch, best_jac, best_model))
        file.close()




if __name__ == '__main__':
    main()