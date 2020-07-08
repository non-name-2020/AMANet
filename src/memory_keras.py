#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.layers import Layer
from keras.layers import GlobalAveragePooling1D, Dense
from keras.models import Input, Model

try:
    from utils import pairwise_add
except ImportError:
    from .utils import pairwise_add


class Memory(Layer):
    def __init__(self, words_num=256, word_size=64, read_heads=4, batch_size=1, **kwargs):
        """
        constructs a memory matrix with read heads and a write head
        Parameters:
        ----------
        words_num: int
            the maximum number of words that can be stored in the memory at the
            same time
        word_size: int
            the size of the individual word in the memory
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
        """

        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size
        self.interface_vector_size = self.word_size * self.read_heads  # R read keys
        self.interface_vector_size += 3 * self.word_size  # 1 write key, 1 erase, 1 content
        self.interface_vector_size += 5 * self.read_heads  # R read key strengths, R free gates, 3xR read modes (each mode for each read has 3 values)
        self.interface_vector_size += 3  # 1 write strength, 1 allocation gate, 1 write gate
        # print(self.interface_vector_size)
        # a words_num x words_num identity matrix
        self.I = tf.constant(np.identity(words_num, dtype=np.float32))  # to support calculate link matrix

        # maps the indecies from the 2D array of free list per batch to
        # their corresponding values in the flat 1D array of ordered_allocation_weighting --> vector a --> need to be sorted
        self.index_mapper = tf.constant(
            np.cumsum([0] + [words_num] * (batch_size - 1), dtype=np.int32)[:, np.newaxis]
            # [[0], [word_num], [word_num*2], [word_num*3], ...]
        )
        self.memory_matrix = tf.fill([self.batch_size, self.words_num, self.word_size], 1e-6) # initial memory matrix
        self.usage_vector = tf.zeros([self.batch_size, self.words_num])  # initial usage vector u
        self.precedence_vector = tf.zeros([self.batch_size, self.words_num])  # initial precedence vector p
        self.link_matrix = tf.zeros([self.batch_size, self.words_num, self.words_num])  # initial link matrix L
        self.write_weighting = tf.fill([self.batch_size, self.words_num], 1e-6)  # initial write weighting
        self.read_weightings = tf.fill([self.batch_size, self.words_num, self.read_heads], 1e-6)  # initial read weightings
        self.read_vectors = tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6)  # initial read vectors
        super(Memory, self).__init__(**kwargs)

    def build(self, input_shape):
        self.interface_weights = self.add_weight(name='interface_weights',
                                                 shape=(input_shape[-1], self.interface_vector_size),
                                                 initializer='glorot_uniform',
                                                 trainable=True)
        super(Memory, self).build(input_shape)

    def call(self, x):
        interface = K.dot(x, self.interface_weights)
        interface = GlobalAveragePooling1D()(interface)
        interface = self.parse_interface_vector(interface)
        write_key = interface['write_key']
        write_strength = interface['write_strength']
        free_gates = interface['free_gates']
        allocation_gate = interface['allocation_gate']
        write_gate = interface['write_gate']
        write_vector = interface['write_vector']
        erase_vector = interface['erase_vector']
        self.read(interface['read_keys'], interface['read_strengths'], interface['read_modes'])
        self.write(write_key, write_strength, free_gates, allocation_gate, write_gate, write_vector, erase_vector)
        return tf.reshape(self.read_vectors,(self.batch_size, self.word_size*self.read_heads))

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.word_size*self.read_heads

    def write(self, key, strength, free_gates, allocation_gate, write_gate, write_vector, erase_vector):
        """
        defines the complete pipeline of writing to memory given the write variables
        and the memory_matrix, usage_vector, link_matrix, and precedence_vector from
        previous step
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix from previous step
        usage_vector: Tensor (batch_size, words_num)
            the usage_vector from the last time step
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read_weightings from the last time step
        write_weighting: Tensor (batch_size, words_num)
            the write_weighting from the last time step
        precedence_vector: Tensor (batch_size, words_num)
            the precedence vector from the last time step
        link_matrix: Tensor (batch_size, words_num, words_num)
            the link_matrix from previous step
        key: Tensor (batch_size, word_size, 1)
            the key to query the memory location with
        strength: (batch_size, 1)
            the strength of the query key
        free_gates: Tensor (batch_size, read_heads)
            the degree to which location at read haeds will be freed
        allocation_gate: (batch_size, 1)
            the fraction of writing that is being allocated in a new locatio
        write_gate: (batch_size, 1)
            the amount of information to be written to memory
        write_vector: Tensor (batch_size, word_size)
            specifications of what to write to memory
        erase_vector: Tensor(batch_size, word_size)
            specifications of what to erase from memory
        Returns : Tuple
            the updated usage vector: Tensor (batch_size, words_num)
            the updated write_weighting: Tensor(batch_size, words_num)
            the updated memory_matrix: Tensor (batch_size, words_num, words_size)
            the updated link matrix: Tensor(batch_size, words_num, words_num)
            the updated precedence vector: Tensor (batch_size, words_num)
        """

        lookup_weighting = self.get_lookup_weighting(key, strength)
        self.update_usage_vector(free_gates)
        sorted_usage, free_list = tf.nn.top_k(-1 * self.usage_vector, self.words_num)  # make it from min to max
        sorted_usage = -1 * sorted_usage  # convert to normal values

        allocation_weighting = self.get_allocation_weighting(sorted_usage, free_list)
        self.update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate)
        self.update_memory(write_vector, erase_vector)
        self.update_link_matrix()
        self.update_precedence_vector()

    def read(self, keys, strengths, read_modes):
        """
        defines the complete pipeline for reading from memory
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the updated memory matrix from the last writing
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read weightings form the last time step
        keys: Tensor (batch_size, word_size, read_heads)
            the kyes to query the memory locations with
        strengths: Tensor (batch_size, read_heads)
            the strength of each read key
        link_matrix: Tensor (batch_size, words_num, words_num)
            the updated link matrix from the last writing
        read_modes: Tensor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes
        Returns: Tuple
            the updated read_weightings: Tensor(batch_size, words_num, read_heads)
            the recently read vectors: Tensor (batch_size, word_size, read_heads)
        """

        lookup_weighting = self.get_lookup_weighting(keys, strengths)  # content weight: later use to produce read weight

        # need last read weights to infer forward, backward --> just mul with link matrix
        forward_weighting, backward_weighting = self.get_directional_weightings()
        self.update_read_weightings(lookup_weighting, forward_weighting, backward_weighting, read_modes)
        self.update_read_vectors()

        return self.read_vectors

    def parse_interface_vector(self, interface_vector):
        parsed = {}
        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.word_size
        erase_end = w_key_end + 1 + self.word_size
        write_end = erase_end + self.word_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.word_size, 1)
        write_shape = erase_shape = (-1, self.word_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        # batch x N x R
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape)
        # batch x R
        parsed['read_strengths'] = tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape)
        # batch x N x 1 --> share similarity function with read
        parsed['write_key'] = tf.reshape(interface_vector[:, r_strengths_end:w_key_end], w_key_shape)
        # batch x 1
        parsed['write_strength'] = tf.reshape(interface_vector[:, w_key_end], (-1, 1))
        # batch x N
        parsed['erase_vector'] = tf.reshape(interface_vector[:, w_key_end + 1:erase_end], erase_shape)
        # batch x N
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)
        # batch x R
        parsed['free_gates'] = tf.reshape(interface_vector[:, write_end:free_end], free_shape)
        # batch x 1
        parsed['allocation_gate'] = tf.expand_dims(interface_vector[:, free_end], 1)
        # batch x 1
        parsed['write_gate'] = tf.expand_dims(interface_vector[:, free_end + 1], 1)
        # batch x 3 x R
        parsed['read_modes'] = tf.reshape(interface_vector[:, free_end + 2:], modes_shape)

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
        parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)

        return parsed

    def get_lookup_weighting(self, keys, strengths):
        """
        retrives a content-based adderssing weighting given the keys
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix to lookup in
        keys: Tensor (batch_size, word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key
        Returns: Tensor (batch_size, words_num, number_of_keys)
            The list of lookup weightings for each provided key
        """
        normalized_memory = tf.nn.l2_normalize(self.memory_matrix, axis=2)  # M=M/|M|

        normalized_keys = tf.nn.l2_normalize(keys, axis=1)  # k=k/|k|
        similiarity = tf.matmul(normalized_memory, normalized_keys)  # cosine sim: (batch_size, word_num, number_of_keys)
        strengths = tf.expand_dims(strengths, 1)  # (batch_size, 1, number_of_keys)

        # (batch_size, word_num, number_of_keys) --softmax on 1-->(batch_size, word_num, number_of_keys)
        return tf.nn.softmax(similiarity * strengths, 1)  # each batch, every row of mem is multiplied with strength and then softmax

    def update_usage_vector(self, free_gates):
        """
        updates and returns the usgae vector given the values of the free gates
        and the usage_vector, read_weightings, write_weighting from previous step
        Parameters:
        ----------
        usage_vector: Tensor (batch_size, words_num)
        read_weightings: Tensor (batch_size, words_num, read_heads)
        write_weighting: Tensor (batch_size, words_num)
        free_gates: Tensor (batch_size, read_heads, )
        Returns: Tensor (batch_size, words_num, )
            the updated usage vector
        """
        free_gates = tf.expand_dims(free_gates, 1)  # (batch_size, 1, read_heads )

        retention_vector = tf.reduce_prod(1 - self.read_weightings * free_gates, 2)  # (batch_size, word_num)
        self.usage_vector = (self.usage_vector + self.write_weighting - self.usage_vector * self.write_weighting) * retention_vector


    def get_directional_weightings(self):
        """
        computes and returns the forward and backward reading weightings
        given the read_weightings from the previous step
        Parameters:
        ----------
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read weightings from the last time step
        link_matrix: Tensor (batch_size, words_num, words_num)
            the temporal link matrix
        Returns: Tuple
            forward weighting: Tensor (batch_size, words_num, read_heads),
            backward weighting: Tensor (batch_size, words_num, read_heads)
        """

        # if your last reading location is i, forward lead you to the next location that is written after i (current write j)
        forward_weighting = tf.matmul(self.link_matrix, self.read_weightings)
        # if your last reading location is i, backward lead you to the previous location that is written before i (last write k)
        backward_weighting = tf.matmul(self.link_matrix, self.read_weightings, transpose_a=True)  # tranpose link and mul
        return forward_weighting, backward_weighting

    def get_allocation_weighting(self, sorted_usage, free_list):
        """
        retreives the writing allocation weighting based on the usage free list
        Parameters:
        ----------
        sorted_usage: Tensor (batch_size, words_num, )
            the usage vector sorted ascendingly
        free_list: Tensor (batch, words_num, )
            the original indecies of the sorted usage vector: free_list[0] = the least use location --> calculated by sorting usage vector
        Returns: Tensor (batch_size, words_num, )
            the allocation weighting for each word in memory
        """
        # cum product makes the first index of result (correspond to less usage one) has bigger value --> should be allocate
        shifted_cumprod = tf.cumprod(sorted_usage, axis=1, exclusive=True)
        # multiply with this even make larger for less usage ones
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod  # batch_size x words_num, the first element is weight for least use

        mapped_free_list = free_list + self.index_mapper  # boardcast add with the offset correspond to batch id
        flat_unordered_allocation_weighting = tf.reshape(unordered_allocation_weighting, (-1,))  # flatten
        flat_mapped_free_list = tf.reshape(mapped_free_list, (-1,))  # flatten
        flat_container = tf.TensorArray(tf.float32, self.batch_size * self.words_num)

        # fill the weights to the original locations
        flat_ordered_weightings = flat_container.scatter(
            flat_mapped_free_list,
            flat_unordered_allocation_weighting
        )

        packed_wightings = flat_ordered_weightings.stack()
        return tf.reshape(packed_wightings, (self.batch_size, self.words_num))

    def update_write_weighting(self, lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        """
        updates and returns the current write_weighting
        Parameters:
        ----------
        lookup_weighting: Tensor (batch_size, words_num, 1)
            the weight of the lookup operation in writing --> diff from one in reading
        allocation_weighting: Tensor (batch_size, words_num)
            the weight of the allocation operation in writing
        write_gate: (batch_size, 1)
            the fraction of writing to be done
        allocation_gate: (batch_size, 1)
            the fraction of allocation to be done
        Returns: Tensor (batch_size, words_num)
            the updated write_weighting
        """

        # remove the dimension of 1 from the lookup_weighting (the third dim, because num write head =1)
        lookup_weighting = tf.squeeze(lookup_weighting, axis=2)

        # the write gate is the final decisor may help protect memory despite other factors
        # allocation gate is computed based on usage
        # allocation gate interpolate between usage and content lookup
        self.write_weighting = write_gate * (
                allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting)


    def update_memory(self, write_vector, erase_vector):
        """
        updates and returns the memory matrix given the weighting, write and erase vectors
        and the memory matrix from previous step
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix from previous step
        write_weighting: Tensor (batch_size, words_num)
            the weight of writing at each memory location
        write_vector: Tensor (batch_size, word_size)
            a vector specifying what to write
        erase_vector: Tensor (batch_size, word_size)
            a vector specifying what to erase from memory
        Returns: Tensor (batch_size, words_num, word_size)
            the updated memory matrix
        """

        # expand data with a dimension of 1 at multiplication-adjacent location
        # to force matmul to behave as an outer product
        write_weighting = tf.expand_dims(self.write_weighting, 2)  # (batch_size, words_num, 1)
        write_vector = tf.expand_dims(write_vector, 1)  # (batch_size, 1, word_size)
        erase_vector = tf.expand_dims(erase_vector, 1)  # (batch_size, 1, word_size)

        # weight and erase are out product to create a matrix erase
        # erase value is reflected differently in each location by the weight
        erasing = self.memory_matrix * (1 - tf.matmul(write_weighting, erase_vector))  # (batch_size, words_num, word_size)
        writing = tf.matmul(write_weighting, write_vector)  # (batch_size, words_num, word_size)
        self.memory_matrix = erasing + writing  # (batch_size, words_num, word_size)


    def update_precedence_vector(self):
        """
        updates the precedence vector given the latest write weighting --> contain info of writting information
        and the precedence_vector from last step
        Parameters:
        ----------
        precedence_vector: Tensor (batch_size. words_num)
            the precedence vector from the last time step
        write_weighting: Tensor (batch_size,words_num)
            the latest write weighting for the memory
        Returns: Tensor (batch_size, words_num)
            the updated precedence vector
        """

        # if current write is to full memory --> no need to refer to writing information from the past--> like write weight
        reset_factor = 1 - tf.reduce_sum(self.write_weighting, 1, keepdims=True)
        self.precedence_vector = reset_factor * self.precedence_vector + self.write_weighting

    def update_link_matrix(self):
        """
        updates and returns the temporal link matrix for the latest write
        given the precedence vector and the link matrix from previous step
        Parameters:
        ----------
        precedence_vector: Tensor (batch_size, words_num)
            the precedence vector from the last time step
        link_matrix: Tensor (batch_size, words_num, words_num)
            the link matrix form the last step
        write_weighting: Tensor (batch_size, words_num)
            the latest write_weighting for the memory
        Returns: Tensor (batch_size, words_num, words_num)
            the updated temporal link matrix
        """

        write_weighting = tf.expand_dims(self.write_weighting, 2)  # (batch_size, words_num, 1)
        precedence_vector = tf.expand_dims(self.precedence_vector, 1)  # (batch_size, 1, words_num)

        # remove old link between all i and j because now we have new weight write
        reset_factor = 1 - pairwise_add(write_weighting, is_batch=True)  # (batch_size, words_num, 1) matrix[i,j]=1-w[i]-w[j]

        # add current link between last write (precedence vector) and cur write weight
        updated_link_matrix = reset_factor * self.link_matrix + tf.matmul(write_weighting,
                                                                          precedence_vector)  # (batch_size, words_num, words_num)

        # diagnoal position should be 0
        self.link_matrix = (1 - self.I) * updated_link_matrix  # eliminates self-links




    def update_read_weightings(self, lookup_weightings, forward_weighting, backward_weighting, read_mode):
        """
        updates and returns the current read_weightings
        Parameters:
        ----------
        lookup_weightings: Tensor (batch_size, words_num, read_heads)
            the content-based read weighting
        forward_weighting: Tensor (batch_size, words_num, read_heads)
            the forward direction read weighting
        backward_weighting: Tensor (batch_size, words_num, read_heads)
            the backward direction read weighting
        read_mode: Tesnor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes
        Returns: Tensor (batch_size, words_num, read_heads)
        """

        # interpolate 3 component: backward forward content
        backward_mode = tf.expand_dims(read_mode[:, 0, :], 1) * backward_weighting
        lookup_mode = tf.expand_dims(read_mode[:, 1, :], 1) * lookup_weightings
        forward_mode = tf.expand_dims(read_mode[:, 2, :], 1) * forward_weighting
        self.read_weightings = backward_mode + lookup_mode + forward_mode


    def update_read_vectors(self):
        """
        reads, updates, and returns the read vectors of the recently updated memory
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the recently updated memory matrix
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the amount of info to read from each memory location by each read head
        Returns: Tensor (word_size, read_heads)
        """

        # the read values
        self.read_vectors = tf.matmul(self.memory_matrix, self.read_weightings, transpose_a=True)


if __name__ == '__main__':
    inputs = Input(batch_shape=(1, None, 256))
    x = Memory()(inputs)
    x = Dense(1, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        # optimizer = Adam(lr = config["lr"], decay = config["lr_d"]),
        metrics=["accuracy"])
    model.summary()

    xs = []
    ys = []
    for i in range(10):
        k = np.random.randint(1,10)
        xs.append(np.random.rand(1,k,256))
        ys.append(np.random.randint(0, 2, size=(1,1)))

    for i in range(len(xs)):
        res = model.train_on_batch(xs[i], ys[i])
        print(res)

