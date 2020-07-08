#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Input, Model
from keras.layers import Embedding, Concatenate, Dense, GlobalAveragePooling1D
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy

try:
    from positionembedding_keras import PositionEmbedding
    from historyattention_keras import HistoryAttention
    from selfattention_keras import SelfAttention
    from memory_keras import Memory
except ImportError:
    from .positionembedding_keras import PositionEmbedding
    from .historyattention_keras import HistoryAttention
    from .selfattention_keras import SelfAttention
    from .memory_keras import Memory

# Calculates the jaccard distance
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Calculates the jaccard similarity
def jaccard(y_true, y_pred):
    y_true_new = K.round(K.clip(y_true, 0, 1))
    y_pred_new = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true_new * y_pred_new)
    union = K.sum(K.clip(y_true_new + y_pred_new, 0, 1))
    return intersection/union


# Calculates the precision
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Calculates the recall
def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


# Calculates the f1-measure, the harmonic mean of precision and recall.
def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    bin_sizes = -(pfas[1:]-pfas[:-1])
    s = ptas * bin_sizes
    return K.sum(s, axis=0)


# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def focal_loss(gamma=2, alpha=0.6):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred = K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def build(config):
    input1 = Input(batch_shape=(1, None,))
    input2 = Input(batch_shape=(1, None,))

    if config["has_history"] and config["has_interaction"]:
        history1 = Input(batch_shape=(1, None, config["feature_size"]*2))
        history2 = Input(batch_shape=(1, None, config["feature_size"]*2))
    elif config["has_history"]:
        history1 = Input(batch_shape=(1, None, config["feature_size"]))
        history2 = Input(batch_shape=(1, None, config["feature_size"]))

    x1 = Embedding(config["vocab_size1"], config["embed_size"],
                    trainable=config["embed_trainable"])(input1)
    if config["has_position_embed"]:
        x1 = PositionEmbedding(size=config["position_embed_size"],
                               mode=config["position_embed_mode"])(x1)
    # output: [batch_size, time_step, nb_head*size_per_head]
    x11 = SelfAttention(config["self_attention_num_heads"], config["self_attention_units"])([x1, x1, x1])
    if config["has_memory"]:
        read_vectors1 = Memory(words_num=config["memory_word_num"],
                               word_size=config["memory_word_size"],
                               read_heads=config["memory_read_heads"])(x11)

    if config["has_history"]:
        weight_history1 = HistoryAttention(bias=False)(history1)

    x2 = Embedding(config["vocab_size2"], config["embed_size"],
                   trainable=config["embed_trainable"])(input2)
    if config["has_position_embed"]:
        x2 = PositionEmbedding(size=config["position_embed_size"],
                               mode=config["position_embed_mode"])(x2)

    # output: [batch_size, time_step, nb_head*size_per_head]
    x22 = SelfAttention(config["self_attention_num_heads"], config["self_attention_units"])([x2, x2, x2])
    if config["has_memory"]:
        read_vectors2 = Memory(words_num=config["memory_word_num"],
                               word_size=config["memory_word_size"],
                               read_heads=config["memory_read_heads"])(x22)
    if config["has_interaction"]:
        x12 = SelfAttention(config["self_attention_num_heads"], config["self_attention_units"])([x1, x2, x2])

        x21 = SelfAttention(config["self_attention_num_heads"], config["self_attention_units"])([x2, x1, x1])

    if config["has_interaction"]:
        x11 = GlobalAveragePooling1D(name='feature11')(x11)
        x22 = GlobalAveragePooling1D(name='feature22')(x22)
        x12 = GlobalAveragePooling1D(name='feature12')(x12)
        x21 = GlobalAveragePooling1D(name='feature21')(x21)

        feature1 = Concatenate(name='feature1')([x11, x12])
        feature2 = Concatenate(name='feature2')([x22, x21])
    else:
        feature1 = GlobalAveragePooling1D(name='feature1')(x11)
        feature2 = GlobalAveragePooling1D(name='feature2')(x22)

    if config["has_history"]:
        weight_history2 = HistoryAttention(bias=False)(history2)

    if config["has_history"]:
        if config["has_memory"]:
            concat_x = Concatenate()([feature1, feature2,
                                      weight_history1, weight_history2,
                                      read_vectors1, read_vectors2])
        else:
            concat_x = Concatenate()([feature1, feature2,
                                      weight_history1, weight_history2])
    else:
        if config["has_memory"]:
            concat_x = Concatenate()([feature1, feature2,
                                      read_vectors1, read_vectors2])
        else:
            concat_x = Concatenate()([feature1, feature2])

    if "softmax" in config and config["softmax"]:
        x = Dense(config["output_size"], activation='softmax', name='output')(concat_x)
    else:
        x = Dense(config["output_size"], activation='sigmoid', name='output')(concat_x)


    if config["has_history"]:
        model = Model(inputs=[input1, input2, history1, history2], outputs=x)
    else:
        model = Model(inputs=[input1, input2], outputs=x)

    if config["optimizer"] == 'sgd':
        optimizer = SGD(lr=config["lr"], decay=config["lr_decay"], momentum=0.9, nesterov=True)
    else:
        optimizer = Adam(lr=config["lr"], decay=config["lr_decay"])
    if config["multi"]:
        if config["focal_loss"]:
            model.compile(
                loss=[focal_loss(gamma=config["focal_loss_gamma"],alpha=config["focal_loss_alpha"])],
                optimizer=optimizer,
                metrics=[binary_accuracy, jaccard, precision, recall, fmeasure, auc])
        else:
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[binary_accuracy, jaccard, precision, recall, fmeasure, auc])
        output_layers = ['feature1', 'feature2', 'output']
    elif "softmax" in config and config["softmax"]:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer)
        output_layers = ['output']
    else:
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer)
        output_layers = ['output']
    model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]

    return model
