import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import math_ops
import numpy as np


def my_dropout(x, keep_prob, scale=None, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name, "dropout", [x]):
        x = tf.convert_to_tensor(x, name="x")
        keep_prob = tf.convert_to_tensor(keep_prob,
                                         dtype=x.dtype,
                                         name="keep_prob")

        noise_shape_gen = noise_shape if noise_shape is not None else tf.shape(x)
        if scale:
            noise_shape_gen = noise_shape_gen / [1, scale, scale, 1]

        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape_gen,
                                           seed=seed,
                                           dtype=x.dtype)
        if scale:
            random_tensor = tf.image.resize_nearest_neighbor(random_tensor, noise_shape[1:3])
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return ret, binary_tensor


def lrelu(x, leak=0.1):
    return tf.maximum(x, leak * x)


def up_conv2d(net, num_outputs, scope, factor=2, resize_fun=tf.image.ResizeMethod.BILINEAR):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, (factor * in_shape[1], factor * in_shape[2]), resize_fun)
    net = slim.conv2d(net, num_outputs=num_outputs, scope=scope, stride=1)
    return net


def pixel_dropout_avg(net, p, pool_size=(3, 3), kernel=None):
    input_shape = net.get_shape().as_list()
    in_avg = slim.avg_pool2d(net, pool_size, stride=1, padding='SAME')
    noise_shape = np.array([input_shape[0], input_shape[1], input_shape[2], 1], dtype=np.int64)
    net, binary_tensor = my_dropout(net, p, kernel, noise_shape=noise_shape, name='pixel_dropout')
    return net + (1.0 - binary_tensor) * in_avg, binary_tensor


def pixel_dropout_global_avg(net, p, kernel=None):
    input_shape = net.get_shape().as_list()
    in_avg = slim.avg_pool2d(net, input_shape[1], stride=1, padding='VALID')
    in_avg = tf.tile(in_avg, [1, input_shape[1], input_shape[2], 1])
    noise_shape = np.array([input_shape[0], input_shape[1], input_shape[2], 1], dtype=np.int64)
    net, binary_tensor = my_dropout(net, p, kernel, noise_shape=noise_shape, name='pixel_dropout')
    return net + (1.0 - binary_tensor) * in_avg, binary_tensor


def conv_group(net, num_out, kernel_size, scope):
    input_groups = tf.split(axis=3, num_or_size_splits=2, value=net)
    output_groups = [slim.conv2d(j, num_out / 2, kernel_size=kernel_size, scope='{}/conv_{}'.format(scope, idx))
                     for (idx, j) in enumerate(input_groups)]
    return tf.concat(axis=3, values=output_groups)


def conv_group_nobn(net, num_out, kernel_size, scope):
    input_groups = tf.split(axis=3, num_or_size_splits=2, value=net)
    output_groups = [slim.conv2d(j, num_out / 2, kernel_size=kernel_size, normalizer_fn=None,
                                 scope='{}/conv_{}'.format(scope, idx))
                     for (idx, j) in enumerate(input_groups)]
    return tf.concat(axis=3, values=output_groups)


def repair_res_layer(inputs, mask, depth, idx, activation_fn, scope=None):
    with slim.variable_scope.variable_scope(scope):
        shortcut = inputs
        if idx == 0:
            preact = inputs
        else:
            preact = slim.batch_norm(inputs, activation_fn=activation_fn, scope='preact')
        residual = tf.concat([preact, mask], 3)
        residual = slim.conv2d(residual, depth, kernel_size=[3, 3], stride=1, scope='conv1')
        residual = slim.conv2d(residual, depth, kernel_size=[3, 3], stride=1, normalizer_fn=None, activation_fn=None,
                               scope='conv2')
        output = shortcut + (1.0 - mask) * residual
        return output
