import tensorflow as tf

import os
from datetime import datetime
import time
import sys
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import debug as tf_debug

from utils import get_checkpoint_path
from globals import LOG_DIR

slim = tf.contrib.slim


class MultiGPUTrainer:
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0003,
                 end_lr=None, beta1=0.5, num_gpus=1, dataset_id=None, data_keys=['image']):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model = model
        self.dataset = dataset
        self.pre_processor = pre_processor
        self.num_epochs = num_epochs
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.end_lr = end_lr if end_lr is not None else 0.01 * init_lr
        self.beta1 = beta1
        self.num_gpus = num_gpus
        self.dataset_id = dataset_id
        self.data_keys = data_keys

        self.summaries = []
        self.moving_avgs_decay = 0.9999
        self.additional_info = None
        self.num_train_steps = None
        self.global_step = None
        self.train_set = None
        self.debug = False

    def get_save_dir(self):
        fname = '{}_{}'.format(self.model.name, self.dataset.name)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        lr = self.learning_rate()
        opts = {'adam': tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1, epsilon=1e-5),
                'sgd': tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {'const': self.init_lr,
                    'alex': self.learning_rate_alex(),
                    'linear': self.learning_rate_linear()}
        return policies[self.lr_policy]

    def get_data_queue(self):
        # Get the training dataset
        if self.dataset_id:
            self.train_set = self.dataset.get_split(self.dataset_id)
            self.num_train_steps = \
                (self.dataset.get_num_dataset(self.dataset_id) / self.model.batch_size) * self.num_epochs
        self.num_train_steps /= self.num_gpus
        print('Number of training steps: {}'.format(self.num_train_steps))
        provider = slim.dataset_data_provider.DatasetDataProvider(self.train_set, num_readers=8,
                                                                  common_queue_capacity=20 * self.model.batch_size,
                                                                  common_queue_min=10 * self.model.batch_size)

        # Parse a serialized Example proto to extract the image and metadata.
        data_list = provider.get(self.data_keys)
        if len(data_list) > 1:
            data_list[1] -= self.dataset.label_offset

        # Pre-process data
        data_list[0] = self.pre_processor.process_train(data_list[0])

        # Make batches
        data_list = tf.train.batch(data_list,
                                   batch_size=self.model.batch_size,
                                   num_threads=8,
                                   capacity=5 * self.model.batch_size)
        if not (type(data_list) is list):
            data_list = [data_list]
        batch_queue = slim.prefetch_queue.prefetch_queue(data_list, capacity=2 * self.num_gpus)
        return batch_queue

    def make_summaries(self, grads, layers):
        self.summaries.append(tf.summary.scalar('learning_rate', self.learning_rate()))
        # Variable summaries
        for variable in slim.get_model_variables():
            self.summaries.append(tf.summary.histogram(variable.op.name, variable))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                self.summaries.append(tf.summary.histogram('gradients/' + var.op.name, grad))
        # Add histograms for activation.
        if layers:
            for layer_id, val in layers.iteritems():
                self.summaries.append(tf.summary.histogram('activations/' + layer_id, val))

    def learning_rate_alex(self):
        # Define learning rate schedule
        num_train_steps = self.num_train_steps
        boundaries = [np.int64(num_train_steps * 0.2), np.int64(num_train_steps * 0.4),
                      np.int64(num_train_steps * 0.6), np.int64(num_train_steps * 0.8)]
        values = [self.init_lr, self.init_lr * 250. ** (-1. / 4.), self.init_lr * 250 ** (-2. / 4.),
                  self.init_lr * 250 ** (-3. / 4.), self.init_lr * 250. ** (-1.)]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def learning_rate_linear(self):
        return tf.train.polynomial_decay(self.init_lr, self.global_step, 0.9 * self.num_train_steps,
                                         end_learning_rate=self.end_lr)

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def build_model(self, batch_queue, tower, opt, scope):
        pass

    def make_init_fn(self, chpt_path):
        pass

    def train_model(self, chpt_path):
        print('Restoring from: {}'.format(chpt_path))
        g = tf.Graph()
        with g.as_default():
            with tf.device('/cpu:0'):
                # Init global step
                self.global_step = slim.create_global_step()

                batch_queue = self.get_data_queue()
                opt = self.optimizer()

                # Calculate the gradients for each model tower.
                tower_grads = []
                loss = None
                layers = None
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_{}'.format(i)) as scope:
                                loss, grads, layers = self.build_model(batch_queue, i, opt, scope)
                                tower_grads.append(grads)
                grad = self.average_gradients(tower_grads)

                # Make summaries
                self.make_summaries(grad, layers)

                # Apply the gradients to adjust the shared variables.
                apply_gradient_op = opt.apply_gradients(grad, global_step=self.global_step)

                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(self.moving_avgs_decay, self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())

                # Group all updates to into a single train op.
                apply_gradient_op = tf.group(apply_gradient_op, variables_averages_op)
                train_op = control_flow_ops.with_dependencies([apply_gradient_op], loss)

                # Create a saver.
                saver = tf.train.Saver(tf.global_variables())
                init_fn = self.make_init_fn(chpt_path)

                # Build the summary operation from the last tower summaries.
                summary_op = tf.summary.merge(self.summaries)

                # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

                # Start running operations on the Graph.
                sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False), graph=g)
                sess.run(init)
                prev_ckpt = get_checkpoint_path(self.get_save_dir())
                if prev_ckpt:
                    print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
                    saver.restore(sess, prev_ckpt)
                elif init_fn:
                    init_fn(sess)

                # Start the queue runners.
                tf.train.start_queue_runners(sess=sess)

                summary_writer = tf.summary.FileWriter(self.get_save_dir(), sess.graph)
                init_step = sess.run(self.global_step)
                print('Start training at step: {}'.format(init_step))
                for step in range(init_step, self.num_train_steps):

                    if self.debug:
                        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 50 == 0:
                        num_examples_per_step = self.model.batch_size * self.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / self.num_gpus
                        print('{}: step {}/{}, loss = {} ({} examples/sec; {} sec/batch)'
                              .format(datetime.now(), step, self.num_train_steps, loss_value,
                                      examples_per_sec, sec_per_batch))
                        sys.stdout.flush()

                    if step % 500 == 0:
                        print('Writing summaries...')
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 5000 == 0 or (step + 1) == self.num_train_steps:
                        checkpoint_path = os.path.join(self.get_save_dir(), 'model.ckpt')
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        saver.save(sess, checkpoint_path, global_step=step)
