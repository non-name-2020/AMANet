#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict


import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from models import Retain
from sklearn.utils import shuffle
from utils import llprint, get_n_params
from dataset import load_tax_data
from metrics import metrics_non_multi, roc_auc_non_multi,prc_auc_non_multi

torch.manual_seed(1203)
model_name = 'Retain-tax'
model_save_name = ''

# time
time_str = time.strftime("%Y%m%d%H%M", time.localtime())

EPOCH = 10
LR = 0.0002
TEST = False


# evaluate
def model_eval(model, data_test):
    print("#####################%eval#####################")
    model.eval()
    eval_real_output = []
    eval_pred_output = []
    eval_pred_output_prob = []
    eval_size = data_test.shape[0]
    for sample_index in range(eval_size):
        llprint("\rBatch: %d/%d" % (sample_index + 1, eval_size))
        input, output = prepare_one_sample(data_test, index=sample_index)
        eval_real_output.append(output[0][0])
        pred_output = model(input)
        pred_prob = F.sigmoid(pred_output).detach().cpu().numpy()[0][0]
        eval_pred_output_prob.append(pred_prob)
        if pred_prob >= 0.5:
            eval_pred_output.append(1)
        else:
            eval_pred_output.append(0)
    print('')
    acc, prec, recall, f1 = metrics_non_multi(eval_real_output, eval_pred_output)
    roc_auc = roc_auc_non_multi(eval_real_output, eval_pred_output_prob)
    prauc = prc_auc_non_multi(eval_real_output, eval_pred_output_prob)
    return acc, prec, recall, f1, prauc, roc_auc


# prepare one sample
def prepare_one_sample(df, index):
    sample = df[index:index+1]
    # input seq1
    input_seq1 = eval(sample["sb_val_qcut"].values[0])
    # inout seq2
    input_seq2 = eval(sample["fp_val_qcut"].values[0])
    # output seq
    input = [[input_seq1, input_seq2]]
    output = sample["type"].values[0]
    if output == 'xk':
        o = 0
    else:
        o = 1
    output = [[o]]

    return input, output


def main(datapath='../../../tax-data/records.csv'):
    data_train, data_valid, data_test, voc_size = load_tax_data(data_path=datapath)
    train_size = data_train.shape[0]
    model = Retain(voc_size)
    # model save dir
    model_save_dir = os.path.join("../../../model/tax-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if TEST:
        model.load_state_dict(torch.load(open(os.path.join(model_save_dir, model_save_name), 'rb')))
        acc, prec, recall, f1, prauc, roc_auc = model_eval(model, data_test)
        print('acc: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n' % (acc, prec, recall, f1, prauc, roc_auc))
    else:
        # train log save dir
        log_save_dir = os.path.join("../../../logs/tax-task", model_name, time_str)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

        # eval logs
        file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        file.write("Number of parameters: %d\n" % get_n_params(model))

        optimizer = Adam(model.parameters(), lr=LR)

        history = defaultdict(list)
        best_f1 = 0.0
        best_epoch = 1
        best_model = None

        for epoch in range(EPOCH):
            file.write("Epoch: %d/%d\n" % (epoch + 1, EPOCH))
            llprint('Epoch: %d/%d\n' % (epoch + 1, EPOCH))
            data_train = shuffle(data_train)
            loss_record = []
            start_time = time.time()
            model.train()
            for sample_index in range(train_size):
                llprint("\rBatch %d/%d" % (sample_index + 1, train_size))
                # 获取第index个企业dual序列
                input, output = prepare_one_sample(data_train, index=sample_index)

                target = output

                pred_output = model(input)
                pred_prob = F.sigmoid(pred_output)
                loss = F.binary_cross_entropy_with_logits(pred_prob, torch.FloatTensor(target))
                loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('')
            acc, prec, recall, f1, prauc, roc_auc = model_eval(model, data_test)
            history['acc'].append(acc)
            history['prec'].append(prec)
            history['recall'].append(recall)
            history['f1'].append(f1)
            history['prauc'].append(prauc)
            history['roc_auc'].append(roc_auc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            file.write("spend time to train: %.2f min\n" % elapsed_time)
            file.write("train loss: %.6f\n" % (np.mean(loss_record)))
            file.write("test acc: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n" % (acc, prec, recall, f1, prauc, roc_auc))
            file.write("###############################################################\n")

            print("spend time to train: %.2f min" % elapsed_time)
            print("train loss: %.6f" % (np.mean(loss_record)))
            print("test acc: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f" % (acc, prec, recall, f1, prauc, roc_auc))
            print("###############################################################\n")

            print("Epoch: %d, loss: %.4f, One epoch time: %.2fm, Appro left time: %.2fh\n" % (epoch + 1,
                                                                                                  np.mean(loss_record),
                                                                                                  elapsed_time,
                                                                                                  elapsed_time * (
                                                                                                          EPOCH - epoch - 1)/60))

            model_save_path = os.path.join(model_save_dir, 'model_%d_%s_%.4f.h5' % ((epoch + 1), time_str, f1))
            torch.save(model.state_dict(), open(model_save_path, 'wb'))
            if best_f1 < f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_model = model_save_path
            file.flush()

        dill.dump(history, open(os.path.join(model_save_dir, 'train_history_%s.pkl' % time_str), 'wb'))
        os.rename(best_model, best_model.replace(".h5", "_best.h5"))
        print("train done. best epoch: %d, best: f1: %f, model path: %s" % (best_epoch, best_f1, best_model))
        file.write("train done. best epoch: %d, best: f1: %f, model path: %s\n" % (best_epoch, best_f1, best_model))
        file.close()


if __name__ == '__main__':
    main(datapath='../../../tax-data/records.csv')