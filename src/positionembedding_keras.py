#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from keras.models import Input,Model


class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call1(self, x):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        print(K.eval(position_j))
        print(position_j)
        position_j = K.expand_dims(position_j, 0)
        print(K.eval(position_j))
        print(position_j)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        print(K.eval(position_i))
        print(position_i)
        position_i = K.expand_dims(position_i, 2)
        print(K.eval(position_i))
        print(position_i)
        position_ij = K.dot(position_i, position_j)
        print(K.eval(position_ij))
        print(position_ij)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        print(K.eval(position_ij))
        print(position_ij)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def call(self, x):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)

        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[0, :, 0]), 0) - 1  # K.arange不支持变长，只好用这种方法生成

        position_i = K.expand_dims(position_i, 1)

        position_ij = K.dot(position_i, position_j)

        cos = tf.expand_dims(tf.cos(position_ij), 2)

        sin = tf.expand_dims(tf.sin(position_ij), 2)

        position_ij = tf.concat([cos, sin], 2)

        position_ij = tf.reshape(position_ij, shape=[-1, self.size])

        position_embedding = K.permute_dimensions(K.repeat(position_ij, x.shape[0]), (1,0,2))

        if self.mode == 'sum':
            return position_embedding + x
        elif self.mode == 'concat':
            return K.concatenate([position_embedding, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return input_shape[0], input_shape[1], input_shape[2] + self.size

if __name__ == '__main__':
    x = Input(batch_shape=(1, None, 10))
    y = PositionEmbedding(10)(x)
    model = Model(inputs=x, outputs=y)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        # optimizer = Adam(lr = config["lr"], decay = config["lr_d"]),
        metrics=["accuracy"])
    model.summary()