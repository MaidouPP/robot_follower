#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def _get_variable(name,
                  shape,
                  initializer,
                  dtype='float32',
                  trainable=True):
    """
    my wrapper of tf.get_variable
    """
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=None,
                           trainable=trainable)

def conv(x,
         shape,
         stride):
    """
    my wrapper of tf.nn.conv2d and tf.nn.bias_add together
    shape: height, width, in, out
    """
    filters_in = x.get_shape()[-1]
    filters_out = shape[-1]
    initializer = tf.contrib.layers.xavier_initializer()
    weights = _get_variable('weights',
                            shape=shape,
                            initializer=initializer)
    bias = _get_variable('bias', [filters_out], initializer)
    # bias = tf.get_variable('bias', [filters_out], 'float',
    #                        tf.constant_initializer(0.05, dtype='float'))
    x = tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1],
                     padding='SAME')
    return tf.nn.bias_add(x, bias)

def resnet_block(x,
                 shape,
                 is_training=True):

    filters_out = shape[-1]
    with tf.variable_scope("stage1"):
        # first stage
        shortcut = x
        x = conv(x, shape, [1, 1])
        x = tf.contrib.layers.batch_norm(x,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         scope='bn')
        shortcut2 = x
        x = x + shortcut
        x = tf.nn.relu(x)

    # second stage
    with tf.variable_scope("stage2"):
        x = conv(x, shape, [1, 1])
        x = tf.contrib.layers.batch_norm(x,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         scope='bn')
        x = shortcut2 + x
        x = tf.nn.relu(x)

    return x
