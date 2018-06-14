import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os
import sys

from MultiGPUTrainer import MultiGPUTrainer
from utils import remove_missing, get_variables_to_train, montage_tf, weights_montage, get_checkpoint_path
from globals import LOG_DIR

slim = tf.contrib.slim


class SDNetTrainer(MultiGPUTrainer):
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0003,
                 end_lr=None, beta1=0.5, num_gpus=1, dataset_id=None, weights_summary=True,
                 restore_scopes=('encoder', 'decoder')):
        MultiGPUTrainer.__init__(self, model, dataset, pre_processor, num_epochs, optimizer, lr_policy, init_lr, end_lr,
                                 beta1, num_gpus, dataset_id)
        self.weights_summary = weights_summary
        self.restore_scopes = restore_scopes
        self.train_set = self.dataset.get_trainset_unlabelled()
        self.num_train_steps = (self.dataset.get_num_train_unlabelled() / self.model.batch_size) * self.num_epochs

    def make_init_fn(self, chpt_path):
        if chpt_path is None:
            ae_chpt_dir = os.path.join(LOG_DIR, '{}_{}/'.format(self.model.ae.name, self.dataset.name))
            chpt_path = get_checkpoint_path(ae_chpt_dir)

        var2restore = slim.get_variables_to_restore(include=self.restore_scopes)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        var2restore = remove_missing(var2restore, chpt_path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(chpt_path, var2restore)
        sys.stdout.flush()

        # Create an initial assignment function.
        def init_fn(sess):
            print('Restoring from: {}'.format(chpt_path))
            sess.run(init_assign_op, init_feed_dict)

        return init_fn

    def build_model(self, batch_queue, tower, opt, scope):
        imgs_train = batch_queue.dequeue()
        tf.summary.image('images/train', montage_tf(imgs_train, 2, 8), max_outputs=1)

        # Create the model
        dec_im, dec_pdrop, layers = self.model.net(imgs_train, reuse=True if tower > 0 else None)
        tf.summary.image('images/autoencoder', montage_tf(dec_im, 2, 8), max_outputs=1)
        tf.summary.image('images/generator', montage_tf(dec_pdrop, 2, 8), max_outputs=1)

        # Show the conv_1 filters
        if self.weights_summary:
            with tf.variable_scope('discriminator', reuse=True):
                weights_disc_1 = slim.variable('conv_1/weights')
            tf.summary.image('weights/conv_1', weights_montage(weights_disc_1, 6, 16),
                             max_outputs=1)

        # Compute losses
        disc_loss = self.model.discriminator_loss(scope, tower)
        gen_loss = self.model.generator_loss(imgs_train, scope, tower)
        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            gen_loss = control_flow_ops.with_dependencies([updates], gen_loss)
            disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

        # Calculate the gradients for the batch of data on this tower.
        grads_gen = opt.compute_gradients(gen_loss, get_variables_to_train('generator'))
        grads_disc = opt.compute_gradients(disc_loss, get_variables_to_train('discriminator'))
        grads = grads_gen + grads_disc
        grad_mult = {var.op.name: 2.0 if var.op.name.endswith('biases') else 1.0 for (_, var) in grads}
        print('Gradient multipliers: {}'.format(grad_mult))
        grads = tf.contrib.training.multiply_gradients(grads, grad_mult)
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return disc_loss+gen_loss, grads, layers
