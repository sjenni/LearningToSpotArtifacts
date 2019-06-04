import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import conv_group_nobn, conv_group


def alexnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, w_reg=0.00005,
                     fix_bn=False):
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.975,
        'epsilon': 0.001,
        'center': True,
        'fused': train_bn,
    }
    with slim.arg_scope([slim.conv2d],
                        kernel_size=kernel_size,
                        padding=padding,
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                return arg_sc


class AlexNet:
    def __init__(self, fc_activation=tf.nn.relu, fix_bn=False, pool5=True, pad='SAME'):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation
        self.use_pool5 = pool5
        self.pad = pad
        self.layers = {}
        self.num_layers = 5

    def classify(self, net, num_classes, reuse=None, training=True, scope='fully_connected'):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(alexnet_argscope(activation=self.fc_activation, padding='SAME', training=training)):
                with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(0.0, 0.005)):
                    enc_shape = net.get_shape().as_list()
                    net = slim.conv2d(net, num_classes, kernel_size=enc_shape[1:3], padding='VALID', scope='fc1',
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer())
                    self.layers['fc_1'] = net
                    net = slim.flatten(net)
        return net

    def discriminate(self, net, reuse=None, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(alexnet_argscope(activation=self.fc_activation, padding='SAME', training=training,
                                                 fix_bn=self.fix_bn)):
                self.layers['input'] = net
                net = slim.conv2d(net, 96, kernel_size=[11, 11], stride=4, scope='conv_1', padding=self.pad,
                                  normalizer_fn=None)
                self.layers['conv_1'] = net
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1', padding=self.pad)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = conv_group_nobn(net, 256, kernel_size=[5, 5], scope='conv_2')
                self.layers['conv_2'] = net
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2', padding=self.pad)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3', normalizer_fn=None)
                self.layers['conv_3'] = net
                net = conv_group_nobn(net, 384, kernel_size=[3, 3], scope='conv_4')
                self.layers['conv_4'] = net
                net = conv_group(net, 256, kernel_size=[3, 3], scope='conv_5')
                self.layers['conv_5'] = net
                if self.use_pool5:
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5', padding=self.pad)
                encoded = net

                drop_pred = slim.conv2d(net, 1, kernel_size=[3, 3], padding='SAME', activation_fn=None,
                                        normalizer_fn=None, scope='drop_pred')
                self.layers['drop_pred'] = net
                drop_pred = slim.flatten(drop_pred)

                enc_shape = net.get_shape().as_list()
                net = slim.conv2d(net, 4096, kernel_size=enc_shape[1:3], padding='VALID', scope='fc_1')
                self.layers['fc_1'] = net
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, 2, kernel_size=[1, 1], padding='VALID', activation_fn=None,
                                  normalizer_fn=None, scope='fc_2')
                self.layers['fc_2'] = net
                net = slim.flatten(net)

                return net, drop_pred, encoded
