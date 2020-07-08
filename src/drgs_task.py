#!/usr/bin/env python
# encoding: utf-8

import os
import time
import argparse
import numpy as np

import tensorflow as tf
from sklearn.utils import shuffle


from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

try:
    from dataset import load_drgs_data, prepare_drgs_dual
    from utils import llprint
    from model_keras_new import build
    from metrics import metrics_multi_class, roc_auc_multi_class, prc_auc_multi_class
except ImportError:
    from .dataset import load_drgs_data, prepare_drgs_dual
    from .utils import llprint
    from .model_keras_new import build
    from .metrics import metrics_multi_class, roc_auc_multi_class, prc_auc_multi_class

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="attention_and_memory_augmented_networks")

parser.add_argument('--datapath', type=str, default='../drgs-data/records.csv', help='data path')
parser.add_argument('--run_mode', type=str, default='test', choices=['train','test'], help='run mode')
parser.add_argument('--softmax', type=bool, default=True, choices=[True, False], help='multi class')

parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--no_tensorboard', action='store_true', help='not use tensorboard')

parser.add_argument('--no_embed_trainable', action='store_true', help='embed not trainable')
parser.add_argument('--embed_size', type=int, default=100, help='embed size')

parser.add_argument('--no_position_embed', action='store_true', help='use position embed or not')
parser.add_argument('--position_embed_size', type=int, default=100, help='position embed size')
parser.add_argument('--position_embed_mode', type=str, default='sum', choices=['sum','concat'], help='position embed mode[sum,concat]')

parser.add_argument('--self_attention_units', type=int, default=64, help='self attention units')
parser.add_argument('--self_attention_num_heads', type=int, default=4, help='self attention num heads')

parser.add_argument('--no_history', action='store_true', help='use history attention or not')
parser.add_argument('--no_interaction', action='store_true', help='use interaction attention or not')

parser.add_argument('--no_memory', action='store_true', help='remove memory or not')
parser.add_argument('--memory_word_num', type=int, default=256, help='memory word num')
parser.add_argument('--memory_word_size', type=int, default=64, help='memory word size')
parser.add_argument('--memory_read_heads', type=int, default=4, help='memory read heads')

parser.add_argument('--feature_size', type=int, default=256, help='feature size')
parser.add_argument('--multi', action='store_true', help='multi-label classification or not')

parser.add_argument('--epochs', type=int, default=15, help='epochs')
parser.add_argument('--focal_loss', action='store_false', help='use focal loss')
parser.add_argument('--focal_loss_alpha', type=float, default=0.6, help='focal loss alpha')
parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='focal loss gamma')

parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=1e-6, help='learning rate decay')

parser.add_argument('--model_path', type=str, help='model path')
args = parser.parse_args()

model_name = "AMANet-drgs"
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


CallBack = TensorBoard(log_dir=('tb-logs/drgs-task/%s/%s' %(model_name, time_str)),  # log dir
                       histogram_freq=0,
                       write_graph=True,
                       write_grads=True,
                       write_images=True,
                       embeddings_freq=0,
                       embeddings_layer_names=None,
                       embeddings_metadata=None)

train_names = ['train_loss']
val_names = ["val_acc", "val_prec", "val_recall", "val_f1", "val_prauc", "val_roc_auc"]

def train(config):

    # model save path
    model_save_dir = os.path.join("../model/drgs-task", model_name, time_str)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # log save path
    log_save_dir = os.path.join("../logs/drgs-task", model_name, time_str)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    # load data
    data_train, data_valid, data_test, voc_size = load_drgs_data(config["datapath"])

    print(voc_size)

    # input1 vocab size
    config["vocab_size1"] = voc_size[0]

    # input1 vocab size
    config["vocab_size2"] = voc_size[1]

    # output vocab size
    config["output_size"] = voc_size[2]

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

    best_f1 = 0.0
    best_epoch = 0
    best_model = ""

    # train
    losses = []
    for epoch in range(config["epochs"]):
        # 新一次迭代，打乱训练集
        data_train = shuffle(data_train)
        start_time = time.time()
        llprint("Epoch %d/%d\n" % (epoch + 1, config["epochs"]))
        cur_losses = []
        train_pred_output_prob = []
        train_real_output = []

        file.write("Epoch: %d/%d\n" % ((epoch + 1), config["epochs"]))

        for patient_index in range(train_size):
            llprint("\rBatch %d/%d" % (patient_index + 1, train_size))
            input_vec1, input_vec2, output_vec, o = prepare_drgs_dual(data_train, voc_size[2], index=patient_index)
            train_real_output.append(o)
            res = model.train_on_batch([input_vec1, input_vec2], output_vec)
            cur_losses.append(res[0])
            predicted = np.argmax(res[1], axis=1)
            train_pred_output_prob.append(res[1][0])
            tmp_output = [0] * config["output_size"]
            tmp_output[predicted[0]] = 1
        losses.append(sum(cur_losses)/len(cur_losses))
        print("----------losses-----------")
        print(losses)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        train_real_output = np.array(train_real_output)
        train_pred_output_prob = np.array(train_pred_output_prob)

        train_acc, train_prec, train_recall, train_f1 = metrics_multi_class(train_real_output, train_pred_output_prob)
        train_roc_auc = roc_auc_multi_class(train_real_output, train_pred_output_prob)
        train_prauc = prc_auc_multi_class(train_real_output, train_pred_output_prob)


        if config["use_tensorboard"]:
            train_logs = [sum(losses)/len(losses)]
            write_log(CallBack, train_names, train_logs, epoch+1)

        print('')
        acc, pre, recall, f1, prauc, roc_auc = model_eval(model, data_valid, config)
        if config["use_tensorboard"]:
            val_logs = [acc, pre, recall, f1, prauc, roc_auc]
            write_log(CallBack, val_names, val_logs, epoch+1)

        file.write("spend time to train: %.2f min\n" % elapsed_time)
        file.write("train loss: %f\n" % (sum(losses)/ len(losses)))
        file.write("valid acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f\n" % (acc, pre, recall, f1, prauc, roc_auc))

        print("spend time to train: %.2f min" % elapsed_time)
        print("train loss: %f, acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % ((sum(losses)/ len(losses)), train_acc, train_prec, train_recall, train_f1, train_prauc, train_roc_auc))
        print("valid acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % (acc, pre, recall, f1, prauc, roc_auc))
        model_save_path = os.path.join(model_save_dir, 'model_%d_%s_%.4f.h5' % ((epoch+1), time_str, f1))
        model.save(model_save_path)

        if best_f1 < f1:
            best_f1 = f1
            best_epoch = epoch + 1
            best_model = model_save_path

        acc, pre, recall, f1, prauc, roc_auc = model_eval(model, data_test, config, type="test")
        print("test acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % (acc, pre, recall, f1, prauc, roc_auc))
        file.write("test acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f\n" % (acc, pre, recall, f1, prauc, roc_auc))
        file.write("###############################################################\n")
        print("###############################################################\n")

        file.flush()

    os.rename(best_model, best_model.replace(".h5", "_best.h5"))
    print("train done. best epoch: %d, best: f1: %f, model path: %s" % (best_epoch, best_f1, best_model))
    file.write("train done. best epoch: %d, best: f1: %f, model path: %s\n" % (best_epoch, best_f1, best_model))
    CallBack.on_train_end(None)
    file.close()

# evaluate
def model_eval(model, dataset, config, type="eval"):
    eval_real_output = []
    eval_pred_output_prob = []
    data_size = len(dataset)
    outputs = [model.get_layer('output').output]
    layer_model = Model(inputs=model.input, outputs=outputs)
    print("#####################%s#####################" % type)
    for patient_index in range(data_size):
        llprint("\rBatch: %d/%d" % (patient_index + 1, data_size))

        dual = prepare_drgs_dual(dataset, config['output_size'], index=patient_index)
        input_vec1, input_vec2, output_vec, o = dual

        layer_model_output = layer_model.predict([input_vec1, input_vec2])
        prob = layer_model_output[0]
        eval_real_output.append(o)
        eval_pred_output_prob.append(prob)
    print('')

    eval_real_output = np.array(eval_real_output)
    eval_pred_output_prob = np.array(eval_pred_output_prob)
    print(np.argmax(eval_real_output, axis=1))
    print(np.argmax(eval_pred_output_prob, axis=1))
    acc, prec, recall, f1 = metrics_multi_class(eval_real_output, eval_pred_output_prob)
    roc_auc = roc_auc_multi_class(eval_real_output, eval_pred_output_prob)
    prauc = prc_auc_multi_class(eval_real_output, eval_pred_output_prob)
    return acc, prec, recall, f1, prauc, roc_auc


if __name__ == "__main__":
    print("#####################args#####################")
    print(args)
    config = {
        "datapath": args.datapath,
        "run_mode": args.run_mode,
        "softmax":args.softmax,
        "debug": args.debug,
        "use_tensorboard": not args.no_tensorboard,
        "has_position_embed": not args.no_position_embed,
        "has_memory": not args.no_memory,
        "has_history": False, #not args.no_history,
        "has_interaction": not args.no_interaction,
        "vocab_size1": 3052,
        "vocab_size2": 1177,
        "output_size": 270,
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
        config["datapath"]='../drgs-data/records.csv'
        train(config=config)
    else:
        model = build(config)
        model.load_weights(os.path.abspath(args.model_path))
        data_train, data_valid, data_test, voc_size = load_drgs_data(data_path='../drgs-data/records.csv')
        acc, pre, recall, f1, prauc, roc_auc = model_eval(model, data_test, config, type="test")
        print("test acc: %f, prec: %f, recall: %f, f1: %f, prauc: %f, roc_auc: %f" % (acc, pre, recall, f1, prauc, roc_auc))


