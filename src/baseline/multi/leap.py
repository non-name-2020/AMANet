#!/usr/bin/env python
# encoding: utf-8

import os

import dill
import time
import sys
sys.path.append(".")
sys.path.append("../..")
import random
import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F

from collections import defaultdict
from sklearn.utils import shuffle

from models import Leap
from utils import llprint, sequence_metric, sequence_output_process, get_n_params
from dataset import load_data

torch.manual_seed(1203)
model_name = 'Leap-medical'
model_save_name = ''
time_str = time.strftime("%Y%m%d%H%M", time.localtime())
EPOCH = 15
LR = 0.0002
TEST = False
device = torch.device('cpu:0')

# evaluate
def model_eval(model, data_test, voc_size):
    print("#####################%eval#####################")
    model.eval()
    ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    records = []
    med_cnt = 0
    visit_cnt = 0
    eval_size = len(data_test)
    for step, input in enumerate(data_test):
        llprint('\rBatch: %d/%d' % (step + 1, eval_size))
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output_logits = model(adm)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)
        records.append(y_pred_label)

        adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        roc_auc.append(adm_roc_auc)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    print('')

    print('avg med', med_cnt / visit_cnt)
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

    model = Leap(voc_size)
    model.to(device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join(model_save_dir, model_save_name), 'rb')))
        adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = model_eval(model, data_test, voc_size)
        print('Jaccard: %.4f, ROC_AUC: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n'
              % (adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1))


    else:
        # train log save dir
        log_save_dir = os.path.join("../../../logs/medical-task", model_name, time_str)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

        # eval logs
        file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        file.write("Number of parameters: %d\n" % get_n_params(model))

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
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target))

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
            history['roc_auc'].append(roc_auc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60

            file.write("spend time to train: %.2f min\n" % elapsed_time)
            file.write("train loss: %.6f\n" % (np.mean(loss_record)))
            file.write("test jac: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f\n" % (ja, avg_p, avg_r, avg_f1, prauc, roc_auc))
            file.write("###############################################################\n")

            print("spend time to train: %.2f min" % elapsed_time)
            print("train loss: %.6f" % (np.mean(loss_record)))
            print("test jac: %.4f, prec: %.4f, recall: %.4f, f1: %.4f, prauc: %.4f, roc_auc: %.4f" % (ja, avg_p, avg_r, avg_f1, prauc, roc_auc))
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

def fine_tune(fine_tune_name=''):
    data_train, data_test, data_test, voc_size = load_data()
    ddi_A = dill.load(open('../../data/ddi_A_final.pkl', 'rb'))

    model = Leap(voc_size)
    model.load_state_dict(torch.load(open(os.path.join("res", model_name, fine_tune_name), 'rb')))

    EPOCH = 30
    LR = 0.0001
    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=LR)
    ddi_rate_record = []
    for epoch in range(1):
        loss_record = []
        start_time = time.time()
        random_train_set = [ random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))


                loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            if K_flag:
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)

                end_time = time.time()
                elapsed_time = (end_time - start_time) / 60
                llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                       np.mean(loss_record),
                                                                                                       elapsed_time,
                                                                                                       elapsed_time * (
                                                                                                               EPOCH - epoch - 1) / 60))

                torch.save(model.state_dict(),
                           open(os.path.join('res', model_name, 'fine_Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)),
                                'wb'))
                print('')

    # test
    torch.save(model.state_dict(), open(
        os.path.join('res', model_name, 'final.model'), 'wb'))



if __name__ == '__main__':
    main()