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
import argparse
import numpy as np

from torch.optim import Adam

import torch.nn.functional as F
from collections import defaultdict
from sklearn.utils import shuffle



from models import GAMENet
from utils import llprint, multi_label_metric, ddi_rate_score, get_n_params
from dataset import load_data

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'GAMENet-medical'
model_save_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--model_save_name', type=str, default=model_save_name, help='model save name')
parser.add_argument('--ddi', action='store_false', default=False, help="using ddi")

args = parser.parse_args()
time_str = time.strftime("%Y%m%d%H%M", time.localtime())
model_name = args.model_name
model_save_name = args.model_save_name
EPOCH = 15
#LR = 0.0001
#weight_decay = 1e-6
LR = 0.0005
weight_decay = 5e-5
#LR = 0.0002
#weight_decay = 1e-5
#LR = 0.001
#weight_decay = 1e-4
emb_dim = 100

TEST = args.eval
Neg_Loss = args.ddi
DDI_IN_MEM = args.ddi
TARGET_DDI = 0.05
T = 0.5
decay_weight = 0.85
device = torch.device('cpu:0')
ehr_adj_path = '../../../medical-data/ehr_adj_final.pkl'
ddi_adj_path = '../../../medical-data/ddi_A_final.pkl'

# evaluate
def model_eval(model, data_test, voc_size):
    print("#####################%eval#####################")
    model.eval()
    smm_record = []
    ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt = 0
    visit_cnt = 0
    eval_size = len(data_test)
    for step, input in enumerate(data_test):
        llprint('Batch: %d/%d' % (step + 1, eval_size))
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        roc_auc.append(adm_roc_auc)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    print('')
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    return ddi_rate, np.mean(ja), np.mean(roc_auc), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    # model save dir
    model_save_dir = os.path.join("../../../model/medical-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    data_train, data_valid, data_test, voc, voc_size = load_data(data_path='../../../medical-data/records_final.pkl',
                                                                voc_path='../../../medical-data/voc_final.pkl')
    train_size = len(data_train)

    model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=emb_dim, device=device, ddi_in_memory=DDI_IN_MEM)
    model.to(device=device)
    if TEST:
        model.load_state_dict(torch.load(open(model_save_name, 'rb')))
        adam_ddi_rate, adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = model_eval(model, data_test, voc_size)
        print('DDI Rate: %.4f, Jaccard: %.4f,  ROC_AUC: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n'
              % (adam_ddi_rate, adm_ja, adm_roc_auc, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1))
    else:
        # train log save dir
        log_save_dir = os.path.join("../../../logs/medical-task", model_name, time_str)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

        # eval logs
        file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
        file.write("Number of parameters: %d\n" % get_n_params(model))
        file.write("LR: %f, weight_decay: %f, emb_dim: %d\n" % (LR, weight_decay, emb_dim))

        optimizer = Adam(list(model.parameters()), lr=LR, weight_decay=weight_decay)
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
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                llprint("\rBatch %d/%d" % (step + 1, train_size))
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.1 * loss3
                            prediction_loss_cnt += 1
                        else:
                            global T
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.1 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.1 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record.append(loss.item())
                llprint('\rBatch: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (step + 1, train_size, prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T = T * decay_weight
            print('')

            ddi_rate, ja, roc_auc, prauc, avg_p, avg_r, avg_f1 = model_eval(model, data_test, voc_size)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['roc_auc'].append(roc_auc)
            history['prauc'].append(prauc)

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


if __name__ == '__main__':
    main()
