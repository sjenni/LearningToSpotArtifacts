import tensorflow as tf
import tensorflow.contrib.slim as slim
from AlexNet import AlexNet
from layers import pixel_dropout_avg, repair_res_layer, up_conv2d
from AutoEncoder import ae_argscope
from utils import montage_tf

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def sdnet_argscope(activation=tf.nn.leaky_relu, kernel_size=(3, 3), padding='SAME', training=True, w_reg=0.00005,
                   fix_bn=False):
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.975,
        'epsilon': 0.001,
        'center': True,
        'scale': True,
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
                        weights_initializer=he):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                return arg_sc


class SNet:
    def __init__(self, autoencoder, batch_size, target_shape, activation_fn=tf.nn.leaky_relu, tag='default', fix_bn=False,
                 disc_pad='VALID'):
        self.ae = autoencoder
        self.name = 'SDNet_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.activation = activation_fn
        self.discriminator = AlexNet(fix_bn=fix_bn, pool5=False, pad=disc_pad)
        self.dec_im = self.dec_drop = self.disc_out = self.drop_pred = self.drop_label = self.rec_weights = None

    def net(self, imgs, reuse=None, train=True):
        enc_im = self.ae.encoder(imgs, reuse=reuse, training=False)

        pixel_drop, drop_mask = pixel_dropout_avg(enc_im, 0.3)
        drop_label_fake = slim.flatten(drop_mask)
        tf.summary.image('images/drop_mask', montage_tf(drop_mask, 2, 8), max_outputs=1)

        drop_label_real = tf.ones_like(drop_label_fake)
        self.rec_weights = tf.image.resize_nearest_neighbor(drop_mask, self.im_shape[:2])

        self.dec_im = self.ae.decoder(enc_im, reuse=reuse, training=False)
        self.dec_drop = self.generator(pixel_drop, drop_mask, reuse=reuse, training=train)

        self.drop_label = tf.concat([drop_label_real, drop_label_fake], 0)
        disc_in = self.scale_imgs(tf.concat([self.dec_im, self.dec_drop], 0))

        self.disc_out, self.drop_pred, _ = self.discriminator.discriminate(disc_in, reuse=reuse, training=train)

        return self.dec_im, self.dec_drop, self.discriminator.layers

    def scale_imgs(self, imgs):
        imgs = (imgs * 127.5) - tf.constant([[[[_R_MEAN - 127.5, _G_MEAN - 127.5, _B_MEAN - 127.5]]]],
                                            dtype=tf.float32, shape=(1, 1, 1, 3))
        return imgs

    def labels_real(self):
        labels = tf.concat(
            [tf.ones((self.batch_size,), dtype=tf.int64), tf.zeros((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def labels_fake(self):
        labels = tf.concat(
            [tf.zeros((self.batch_size,), dtype=tf.int64), tf.ones((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def classifier(self, img, num_classes, reuse=None, training=True):
        img = self.scale_imgs(img)
        _, _, model = self.discriminator.discriminate(img, reuse=reuse, training=training)
        model = self.discriminator.classify(model, num_classes, reuse=reuse, training=training)
        return model, self.discriminator.layers

    def generator(self, net, drop_mask, reuse=None, training=True):
        f_dims = DEFAULT_FILTER_DIMS
        num_layers = self.ae.num_layers
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope(sdnet_argscope(activation=self.activation, padding='SAME', training=training)):
                net = repair_res_layer(net, drop_mask, f_dims[num_layers-1], 0, activation_fn=self.activation,
                                       scope='repair_0')

        for l in range(0, num_layers - 1):
            with tf.variable_scope('decoder', reuse=True):
                with slim.arg_scope(ae_argscope(activation=self.activation, padding='SAME', training=False)):
                    net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 2], scope='deconv_{}'.format(l))
                drop_mask = upsample_mask(drop_mask)
            with tf.variable_scope('generator', reuse=reuse):
                with slim.arg_scope(sdnet_argscope(activation=self.activation, padding='SAME', training=training)):
                    net = repair_res_layer(net, drop_mask, f_dims[num_layers - l - 2], 0, activation_fn=self.activation,
                                           scope='repair_{}'.format(l + 1))

        with tf.variable_scope('decoder', reuse=True):
            with slim.arg_scope(ae_argscope(activation=self.activation, padding='SAME', training=False)):
                net = up_conv2d(net, num_outputs=32, scope='deconv_{}'.format(num_layers))
            drop_mask = upsample_mask(drop_mask)

        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope(sdnet_argscope(activation=self.activation, padding='SAME', training=training)):
                net = repair_res_layer(net, drop_mask, 32, 0, activation_fn=self.activation,
                                       scope='repair_{}'.format(num_layers))

        with tf.variable_scope('decoder', reuse=True):
            with slim.arg_scope(ae_argscope(activation=self.activation, padding='SAME', training=False)):
                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(num_layers + 1), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
        return net

    def generator_loss(self, imgs_train, scope, tower):
        fake_loss = tf.losses.softmax_cross_entropy(self.labels_fake(), self.disc_out, scope=scope,
                                                    weights=1.0)
        tf.summary.scalar('losses/generator_{}'.format(tower), fake_loss)
        ae_loss = tf.losses.mean_squared_error(imgs_train, self.dec_drop, scope=scope,
                                               weights=30.0 * self.rec_weights)
        tf.summary.scalar('losses/generator_mse_{}'.format(tower), ae_loss)
        losses_gen = tf.losses.get_losses(scope)
        losses_gen += tf.losses.get_regularization_losses(scope)
        gen_loss = tf.add_n(losses_gen, name='gen_total_loss')
        return gen_loss

    def discriminator_loss(self, scope, tower):
        real_loss = tf.losses.softmax_cross_entropy(self.labels_real(), self.disc_out, scope=scope, weights=1.0)
        tf.summary.scalar('losses/discriminator_{}'.format(tower), real_loss)

        drop_pred_loss = tf.losses.sigmoid_cross_entropy(self.drop_label, self.drop_pred, scope=scope, weights=0.1)
        tf.summary.scalar('losses/drop_pred_{}'.format(tower), drop_pred_loss)

        losses_disc = tf.losses.get_losses(scope)
        losses_disc += tf.losses.get_regularization_losses(scope)
        disc_total_loss = tf.add_n(losses_disc, name='disc_total_loss')

        real_pred = tf.arg_max(self.disc_out, 1)
        real_true = tf.arg_max(self.labels_real(), 1)
        tf.summary.scalar('accuracy/discriminator_{}'.format(tower), slim.metrics.accuracy(real_pred, real_true))

        return disc_total_loss


def upsample_mask(mask):
    mask_size = mask.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(mask, [mask_size[1] * 2, mask_size[2] * 2])