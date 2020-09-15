#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

class MeanAggregator(Layer):
    def __init__(self, units, input_dim, neighbor_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False, seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units, self.neighbor_max, self.concat, self.dropout_rate, self.l2_reg = units, neighbor_max, concat, dropout_rate,l2_reg
        self.use_bias, self.activation, self.seed, self.input_dim = use_bias, activation, seed, input_dim
    def build(self, input_shape):
        self.neighbor_weights = self.add_weight(shape=(self.input_dim, self.units),
                                                initializer=glorot_uniform(seed=self.seed),
                                                regularizer=l2(self.l2_reg), name="neighbor_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(), name="bias_weight")
        self.dropout = Dropout(self.dropout_rate)
        self.built = True
    def call(self, inputs, training=None):
        features, node, neighbors = inputs
        node_feat = tf.nn.embedding_lookup(features, node)
        neighbor_feat = tf.nn.embedding_lookup(features, neighbors)
        node_feat = self.dropout(node_feat, training=training)
        neighbor_feat = self.dropout(neighbor_feat, training=training)
        concat_feat = tf.concat([neighbor_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keep_dims=False)
        output = tf.matmul(concat_mean, self.neighbor_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        output._uses_learning_phase = True
        return output
    def get_config(self):
        config = {'units':self.units, 'concat':self.concat, "seed":self.seed}
        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class PoolingAggregator(Layer):
    def __init__(self, units, input_dim, neighbor_max, aggregator="meanpooling", concat=True, dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False, seed=1024):
        super(PoolingAggregator, self).__init__()
        self.output_dim, self.input_dim, self.concat, self.pooling, self.dropout_rate = units, input_dim, concat, aggregator, dropout_rate
        self.l2_reg, self.use_bias, self.activation, self.neighbor_max, self.seed = l2_reg, use_bias, activation, neighbor_max, seed
    def build(self, input_shape):
        self.dense_layers = [Dense(self.input_dim, activation="relu", use_bias=True, kernel_regularizer=l2(self.l2_reg))]
        self.neighbor_weights = self.add_weight(shape=(self.input_dim*2, self.output_dim),
                                                initializer=glorot_uniform(seed=self.seed),
                                                regularizer=l2(self.l2_reg), name="neighbor_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ), initializer=Zeros(), name="bias_weight")

        self.built = True
    def call(self, inputs, mask=None):
        features, node, neighbors = inputs
        node_feat = tf.nn.embedding_lookup(features, node)
        neighbor_feat = tf.nn.embedding_lookup(features, neighbors)
        dims = tf.shape(neighbor_feat)
        batch_size, num_neighbors = dims[0], dims[1]
        h_reshaped = tf.reshape(neighbor_feat, (batch_size*num_neighbors, self.input_dim))
        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neighbor_feat = tf.reshape(h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))
        if self.pooling == "meanpooling":
            neighbor_feat = tf.reduce_mean(neighbor_feat, axis=1, keep_dims=False)
        else:
            neighbor_feat = tf.reduce_max(neighbor_feat, axis=1)
        output = tf.concat([tf.squeeze(node_feat, axis=1), neighbor_feat], axis=-1)
        output = tf.matmul(output, self.neighbor_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output
    def get_config(self):
        config = {'output_dim': self.output_dim, "concat":self.concat}
        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu, aggregator_type='mean',
              dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]
    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator
    h = features
    for i in range(0, len(neighbor_num)):
        if i>0:
            feature_dim = n_hidden
        if i == len(neighbor_num) -1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neighbor_max=neighbor_num[i],
                       aggregator=aggregator_type)([h, node_input, neighbor_input[i]])
        output = h
        input_list = [features, node_input] + neighbor_input
        model = Model(input_list, outputs=output)
        return model
def sample_neighbors(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 采样邻居节点
    _sample = np.random.choice
    neighbors = [list(G[int(node)]) for node in nodes]  # nodes里每个节点的邻居
    if sample_num:
        if self_loop:
            sample_num -= 1
        samp_neighbors = [list(_sample(neighbor, sample_num, replace=False)) if len(neighbor)>=sample_num
                          else list(_sample(neighbor, sample_num, replace=True)) for neighbor in neighbors]  # 采样邻居
        if self_loop:
            samp_neighbors = [samp_neighbors + list([nodes[i]]) for i, samp_neighbor in enumerate(samp_neighbors)] # gcn邻居要加上自己
        if shuffle:
            samp_neighbors = [list(np.random.permutation(x)) for x in samp_neighbors]
    else:
        samp_neighbors = neighbors
    return np.asarray(samp_neighbors), np.asarray(list(map(len, samp_neighbors)))
