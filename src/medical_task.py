#!/usr/bin/env python
# encoding: utf-8
import os
import time
import argparse
import numpy as np

import tensorflow as tf
import keras.backend as K


from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

try:
    from dataset import load_data, prepare_mimic_sample_dual_persist
    from utils import llprint
    from model_keras_new import build
    from metrics import metrics_multi, roc_auc_multi, precision_auc, prc_auc_multi
except ImportError:
    from .dataset import load_data, prepare_mimic_sample_dual_persist
    from .utils import llprint
    from .model_keras_new import build
    from .metrics import metrics_multi, roc_auc_multi, precision_auc, prc_auc_multi

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="attention_and_memory_augmented_networks")

parser.add_argument('--datapath', type=str, default='../medical-data/records_final.csv', help='data path')
parser.add_argument('--run_mode', type=str, default='test', choices=['train','test'], help='run mode')

parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--no_tensorboard', action='store_true', help='use tensorboard')

parser.add_argument('--no_embed_trainable', action='store_true', help='embed trainable')
parser.add_argument('--embed_size', type=int, default=100, help='embed size')

parser.add_argument('--no_position_embed', action='store_true', help='use position embed or not')
parser.add_argument('--position_embed_size', type=int, default=100, help='position embed size')
parser.add_argument('--position_embed_mode', type=str, default='sum', choices=['sum','concat'], help='position embed mode[sum,concat]')


parser.add_argument('--no_history', action='store_true', help='use history attention or not')
parser.add_argument('--no_interaction', action='store_true', help='use interaction attention or not')

parser.add_argument('--self_attention_units', type=int, default=64, help='self attention units')
parser.add_argument('--self_attention_num_heads', type=int, default=4, help='self attention num heads')

parser.add_argument('--no_memory', action='store_true', help='remove memory or not')
parser.add_argument('--memory_word_num', type=int, default=256, help='memory word num')
parser.add_argument('--memory_word_size', type=int, default=64, help='memory word size')
parser.add_argument('--memory_read_heads', type=int, default=4, help='memory read heads')

parser.add_argument('--feature_size', type=int, default=256, help='feature size')
parser.add_argument('--multi', action='store_false', help='multi-label classification or not')


parser.add_argument('--epochs', type=int, default=15, help='epochs')
parser.add_argument('--focal_loss', action='store_false', help='use focal loss')
parser.add_argument('--focal_loss_alpha', type=float, default=0.6, help='focal loss alpha')
parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='focal loss gamma')


parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=1e-6, help='learning rate decay')



parser.add_argument('--model_path', type=str, help='model path')
args = parser.parse_args()

model_name = "AMANet-medical"
# time
time_str = time.strftime("%Y%m%d%H%M", time.localtime())


def write_log(callback, names, logs, epoch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, epoch_no)
        callback.writer.flush()


CallBack = TensorBoard(log_dir=('../tb-logs/medical-task/%s/%s' %(model_name, time_str)),  # log dir
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

train_names = ['train_loss', 'train_acc', "train_jac", "train_prec", "train_recall", "train_f1", "train_roc_auc"]
val_names = ["val_jac", "val_prec", "val_recall", "val_f1", "val_roc_auc"]


def train(config):
    # model save path
    model_save_dir = os.path.join("../model/medical-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # log save path
    log_save_dir = os.path.join("../logs/medical-task", model_name, time_str)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    # load data
    data_train, data_valid, data_test, voc, voc_size = load_data()

    # input1 vocab size
    config["vocab_size1"] = voc_size[0]

    # input1 vocab size
    config["vocab_size2"] = voc_size[1]

    # output vocab size
    # 0 is padding char, 1 is decoder char, removed
    config["output_size"] = voc_size[2] - 2

    # build model
    model = build(config)

    # plot model graph
    model_graph_file = os.path.join(model_save_dir, ("model_%s.png" % time_str))
    plot_model(model, to_file=model_graph_file)

    # model summary
    model.summary()

    # model tensorboard logs
    CallBack.set_model(model)

    # eval logs
    file = open(os.path.join(log_save_dir, "statistic_%s.txt" % time_str), "w+")
    file.write(str(config)+"\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    train_size = len(data_train)

    best_jaccard = 0.0
    best_epoch = 0
    best_model = ""

    # feature layer output
    if config["has_history"]:
        get_layer_output = K.function(model.input, [model.get_layer('feature1').output, model.get_layer('feature2').output])

    # train
    for epoch in range(config["epochs"]):

        # shuffle
        np.random.shuffle(data_train)
        start_time = time.time()
        llprint("Epoch %d/%d\n" % (epoch + 1, config["epochs"]))
        # batch_size=1
        losses = []
        accs = []
        jacs = []
        precs = []
        recalls = []
        f1s = []
        praucs = []
        roc_aucs = []

        file.write("Epoch: %d/%d\n" % (epoch + 1, config["epochs"]))

        for patient_index in range(train_size):
            llprint("\rBatch %d/%d" % (patient_index + 1, train_size))

            if config["has_history"] and config["has_interaction"]:
                history1_list = [np.zeros((1, config["feature_size"]*2))]
                history2_list = [np.zeros((1, config["feature_size"]*2))]
            elif config["has_history"]:
                history1_list = [np.zeros((1, config["feature_size"]))]
                history2_list = [np.zeros((1, config["feature_size"]))]

            # 获取第patient_index个病人的visits
            adms = prepare_mimic_sample_dual_persist(data_train, config["output_size"], index=patient_index)
            # 每个病人每一次visit
            for adm in adms:

                input_vec1, input_vec2, output_vec, o = adm
                if len(output_vec) == 0 and len(output_vec[0]) == 0:
                    continue

                if config["has_history"]:
                    history1 = np.transpose(np.array(history1_list), (1, 0, 2))
                    history2 = np.transpose(np.array(history2_list), (1, 0, 2))
                    res = model.train_on_batch([input_vec1, input_vec2, history1, history2], np.array(output_vec))

                    layer_model_output = get_layer_output([input_vec1, input_vec2, history1, history2])
                    history1_list.append(layer_model_output[0])
                    history2_list.append(layer_model_output[1])
                else:
                    res = model.train_on_batch([input_vec1, input_vec2], np.array(output_vec))
                # print(res)
                losses.append(res[0])
                accs.append(res[1])
                jacs.append(res[2])
                precs.append(res[3])
                recalls.append(res[4])
                f1s.append(res[5])
                roc_aucs.append(res[6])

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        if config["use_tensorboard"]:
            train_logs = [sum(losses)/len(losses), sum(accs)/len(accs), sum(jacs)/len(jacs), sum(precs)/len(precs), sum(recalls)/len(recalls), sum(f1s)/len(f1s), sum(roc_aucs)/len(roc_aucs)]
            write_log(CallBack, train_names, train_logs, epoch+1)
        print('')
        pre, recall, jac, f1, prauc, roc_auc = model_eval(model, data_valid, config)
        if config["use_tensorboard"]:
            val_logs = [jac, pre, recall, f1, prauc, roc_auc]
            write_log(CallBack, val_names, val_logs, epoch+1)

        file.write("spend time to train: %.2f min\n" % elapsed_time)
        file.write("avg train loss: %f\n" % (sum(losses)/ len(losses)))
        file.write("avg valid jaccard: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f\n" % (jac, pre, recall, f1, prauc, roc_auc))
        file.write("###############################################################\n")

        print("spend time to train: %.2f min" % elapsed_time)
        print("avg train loss: %f" % (sum(losses)/ len(losses)))
        print("avg valid jaccard: %f, prec: %f, recall: %f, f1: %f, prauc: %f,  roc_auc: %f" % (jac, pre, recall, f1, prauc, roc_auc))
        model_save_path = os.path.join(model_save_dir, 'model_%d_%s_%.4f.h5' % (epoch + 1, time_str, jac))
        model.save(model_save_path)
        if best_jaccard < jac:
            best_jaccard = jac
            best_epoch = epoch + 1
            best_model = model_save_path

        pre, recall, jac, f1, prauc, roc_auc = model_eval(model, data_test, config, type="test")
        print("avg test jaccard: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % (jac, pre, recall, f1, prauc, roc_auc))
        file.write("avg test jaccard: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f\n" % (jac, pre, recall, f1, prauc, roc_auc))
        file.write("###############################################################\n")
        print("###############################################################\n")

        file.flush()

    os.rename(best_model, best_model.replace(".h5", "_best.h5"))
    print("train done. best epoch: %d, best: jaccard: %f, model path: %s" % (best_epoch, best_jaccard, best_model))
    file.write("train done. best epoch: %d, best: jaccard: %f, model path: %s\n" % (best_epoch, best_jaccard, best_model))
    CallBack.on_train_end(None)
    file.close()


# evaluate
def model_eval(model, dataset, config, type="eval"):
    pres = []
    recalls = []
    jacs = []
    f1scores = []
    praucs = []
    roc_aucs = []
    data_size = len(dataset)
    if config["has_history"]:
        outputs = [model.get_layer('feature1').output,
                  model.get_layer('feature2').output,
                  model.get_layer('output').output]
    else:
        outputs = [model.get_layer('output').output]
    layer_model = Model(inputs=model.input, outputs=outputs)

    print("#####################%s#####################" % type)
    for patient_index in range(data_size):
        llprint("\rBatch: %d/%d" % (patient_index + 1, data_size))
        if config["has_history"] and config["has_interaction"]:
            history1_list = [np.zeros((1, config["feature_size"]*2))]
            history2_list = [np.zeros((1, config["feature_size"]*2))]
        elif config["has_history"]:
            history1_list = [np.zeros((1, config["feature_size"]))]
            history2_list = [np.zeros((1, config["feature_size"]))]
        # 获取第patient_index个病人的visits
        adms = prepare_mimic_sample_dual_persist(dataset, config["output_size"], index=patient_index)
        # 每个病人每一次visit
        tmp_praucs = []
        tmp_pres = []
        tmp_recalls = []
        tmp_jacs = []
        tmp_f1scores = []
        tmp_praucs = []
        tmp_roc_aucs = []
        for adm in adms:

            input_vec1, input_vec2, output_vec, o = adm
            if len(output_vec) == 0 and len(output_vec[0]) == 0:
                continue

            if config["has_history"]:
                history1 = np.transpose(np.array(history1_list), (1, 0, 2))
                history2 = np.transpose(np.array(history2_list), (1, 0, 2))
                layer_model_output = layer_model.predict([input_vec1, input_vec2, history1, history2])
                history1_list.append(layer_model_output[0])
                history2_list.append(layer_model_output[1])
                out = layer_model_output[2]
                predict = np.argwhere(out >= 0.5)
                predict = predict[:,1]
                roc_auc = roc_auc_multi(output_vec[0], list(out[0,:]))
                prauc = prc_auc_multi(output_vec[0], list(out[0,:]))
            else:
                layer_model_output = layer_model.predict([input_vec1, input_vec2])
                out = layer_model_output[0]
                predict = np.argwhere(out >= 0.5)
                predict = predict[:,0]
                roc_auc = roc_auc_multi(output_vec[0], list(out))
                prauc = prc_auc_multi(output_vec[0], list(out))

            predict_list = [0] * config["output_size"]
            for index in predict:
                predict_list[index] = 1
            prec, recall, jac, f1 = metrics_multi(o, predict)
            tmp_pres.append(prec)
            tmp_recalls.append(recall)
            tmp_jacs.append(jac)
            tmp_f1scores.append(f1)
            tmp_praucs.append(prauc)
            tmp_roc_aucs.append(roc_auc)
        pres.append(np.mean(tmp_pres))
        recalls.append(np.mean(tmp_recalls))
        jacs.append(np.mean(tmp_jacs))
        f1scores.append(np.mean(tmp_f1scores))

        praucs.append(np.mean(tmp_praucs))
        roc_aucs.append(np.mean(tmp_roc_aucs))
    print('')
    return sum(pres)/len(pres), \
           sum(recalls)/len(recalls), \
           sum(jacs)/len(jacs),\
           sum(f1scores)/len(f1scores), \
           sum(praucs)/len(praucs), \
           sum(roc_aucs)/len(roc_aucs)


if __name__ == "__main__":
    print("#####################args#####################")
    print(args)
    config = {
        "run_mode": args.run_mode,
        "debug": args.debug,
        "use_tensorboard": not args.no_tensorboard,
        "has_position_embed": not args.no_position_embed,
        "has_memory": not args.no_memory,
        "has_history": not args.no_history,
        "has_interaction": not args.no_interaction,
        "vocab_size1": 1960,
        "vocab_size2": 1432,
        "output_size": 151,
        "embed_trainable": not args.no_embed_trainable,
        "embed_size": args.embed_size,
        "position_embed_size": args.position_embed_size,
        "position_embed_mode": args.position_embed_mode,
        "self_attention_units": args.self_attention_units,
        "self_attention_num_heads": args.self_attention_num_heads,
        "memory_word_num": args.memory_word_num,
        "memory_word_size": args.memory_word_size,
        "memory_read_heads": args.memory_read_heads,
        "feature_size": args.feature_size,
        "multi": args.multi,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "focal_loss": args.focal_loss,
        "focal_loss_alpha": args.focal_loss_alpha,
        "focal_loss_gamma": args.focal_loss_gamma,
        "lr":args.lr,
        "lr_decay":args.lr_decay
    }
    print("#####################config#####################")
    print(config)
    if config["run_mode"] == "train":
        train(config=config)
    else:
        model = build(config)
        model.load_weights(os.path.abspath(args.model_path))
        data_train, data_valid, data_test, voc, voc_size = load_data()
        pre, recall, jac, f1, prauc, roc_auc = model_eval(model, data_test, config, type="test")
        print("avg test jaccard: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % (jac, pre, recall, f1, prauc, roc_auc))

