import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import up_conv2d

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]


def ae_argscope(activation=tf.nn.elu, kernel_size=(3, 3), padding='SAME', w_reg=0.00005, training=True):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.975,
        'epsilon': 0.001,
        'center': True,
        'fused': training,
    }
    he = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    with slim.arg_scope([slim.conv2d],
                        activation_fn=activation,
                        kernel_size=kernel_size,
                        padding=padding,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_regularizer=slim.l2_regularizer(w_reg),
                        weights_initializer=he) as arg_sc:
        return arg_sc


class AutoEncoder:
    def __init__(self, num_layers, batch_size, target_shape, activation_fn=tf.nn.relu, tag='default'):
        self.name = 'AutoEncoder_{}_{}'.format(num_layers, tag)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.activation = activation_fn
        self.dec_im = None
        self.imgs_train = None

    def net(self, imgs, reuse=None, train=True):
        self.imgs_train = imgs
        enc_im = self.encoder(imgs, reuse=reuse, training=train)
        print('Shape of encoded features: '.format(enc_im.get_shape().as_list()))
        self.dec_im = self.decoder(enc_im, reuse=reuse, training=train)
        return self.dec_im

    def encoder(self, net, reuse=None, training=True):
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(ae_argscope(activation=self.activation, kernel_size=(2, 2), padding='SAME',
                                            training=training)):
                net = slim.conv2d(net, kernel_size=(3, 3), num_outputs=32, stride=1, scope='conv_0')
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))

                return net

    def decoder(self, net, reuse=None, training=True):
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope(ae_argscope(activation=self.activation, padding='SAME', training=training)):
                for l in range(0, self.num_layers - 1):
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - 2], scope='deconv_{}'.format(l))
                net = up_conv2d(net, num_outputs=32, scope='deconv_{}'.format(self.num_layers))

                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(self.num_layers + 1), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net

    def ae_loss(self, scope, tower=0):
        ae_loss = tf.losses.mean_squared_error(self.imgs_train, self.dec_im, scope=scope, weights=30.0)
        tf.summary.scalar('losses/ae_loss_{}'.format(tower), ae_loss)
        losses_ae = tf.losses.get_losses(scope)
        losses_ae += tf.losses.get_regularization_losses(scope)
        ae_total_loss = tf.add_n(losses_ae, name='ae_total_loss_{}'.format(tower))
        return ae_total_loss
