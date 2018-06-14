import tensorflow as tf
from MultiGPUTrainer import MultiGPUTrainer

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
import sys
import os

from utils import assign_from_checkpoint_fn, montage_tf, weights_montage, get_checkpoint_path
from globals import LOG_DIR

slim = tf.contrib.slim


class ClassifierTrainer(MultiGPUTrainer):
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0003,
                 end_lr=None, beta1=0.5, num_gpus=1, num_conv2train=None, num_conv2init=None,
                 dataset_id=None):
        MultiGPUTrainer.__init__(self, model, dataset, pre_processor, num_epochs, optimizer, lr_policy, init_lr, end_lr,
                                 beta1, num_gpus, dataset_id)
        self.data_keys = ['image', 'label']
        self.num_conv2train = num_conv2train
        self.num_conv2init = num_conv2init
        self.additional_info = 'conv_{}'.format(self.num_conv2train)
        self.train_set = self.dataset.get_trainset_labelled()
        self.num_train_steps = (self.dataset.get_num_train_labelled() / self.model.batch_size) * self.num_epochs

    def classification_loss(self, scope, preds_train, labels_train, tower=0):
        labels_train = self.dataset.format_labels(labels_train)

        # Define the loss
        loss = tf.losses.softmax_cross_entropy(labels_train, preds_train, scope=scope)
        tf.summary.scalar('losses/softmax_loss_tower{}'.format(tower), loss)
        train_losses = tf.losses.get_losses(scope)
        train_losses += tf.losses.get_regularization_losses(scope)
        total_train_loss = math_ops.add_n(train_losses, name='total_train_loss')
        tf.summary.scalar('losses/total_loss_tower{}'.format(tower), total_train_loss)

        # Compute accuracy
        predictions = tf.argmax(preds_train, 1)
        tf.summary.scalar('accuracy/training accuracy',
                          slim.metrics.accuracy(predictions, tf.argmax(labels_train, 1)))
        tf.summary.histogram('labels', tf.argmax(labels_train, 1))
        tf.summary.histogram('predictions', predictions)
        return total_train_loss

    def get_variables_to_train(self, num_conv_train):
        var2train = []
        num_disc_layers = self.model.discriminator.num_layers
        for i in range(num_conv_train):
            vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(num_disc_layers - i)],
                                               exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
        vs = slim.get_variables_to_restore(include=['fully_connected'],
                                           exclude=['discriminator/fully_connected'])
        vs = list(set(vs).intersection(tf.trainable_variables()))
        var2train += vs
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        sys.stdout.flush()
        return var2train

    def make_init_fn(self, chpt_path):
        if self.num_conv2init == 0:
            return None
        else:
            if chpt_path is None:
                fname = '{}_{}'.format(self.model.name, self.dataset.name)
                chpt_path = get_checkpoint_path(os.path.join(LOG_DIR, '{}/'.format(fname)))

            # Specify the layers of the model you want to exclude
            var2restore = []
            for i in range(self.num_conv2init):
                vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(i + 1)],
                                                   exclude=['discriminator/fully_connected'])
                var2restore += vs
            init_fn = assign_from_checkpoint_fn(chpt_path, var2restore, ignore_missing_vars=True)
            print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
            sys.stdout.flush()
            return init_fn

    def build_model(self, batch_queue, tower, opt, scope):
        # Get training batches
        imgs_train, labels_train = batch_queue.dequeue()
        tf.summary.image('images/train', montage_tf(imgs_train, 2, 8), max_outputs=1)

        # Get predictions
        preds_train, layers = self.model.classifier(imgs_train, self.dataset.num_classes,
                                                    reuse=True if tower > 0 else None)
        # Show the conv_1 filters
        with tf.variable_scope('discriminator', reuse=True):
            weights_disc_1 = slim.variable('conv_1/weights')
        tf.summary.image('weights/conv_1', weights_montage(weights_disc_1, 6, 16),
                         max_outputs=1)

        # Compute the loss
        loss = self.classification_loss(scope, preds_train, labels_train, tower=tower)
        tf.get_variable_scope().reuse_variables()

        # Handle dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients for the batch of data on this tower.
        grads = opt.compute_gradients(loss, self.get_variables_to_train(self.num_conv2train))
        grad_mult = {var.op.name: 2.0 if var.op.name.endswith('biases') else 1.0 for (_, var) in grads}
        print('Gradient multipliers: {}'.format(grad_mult))
        grads = tf.contrib.training.multiply_gradients(grads, grad_mult)
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss, grads, layers
