import tensorflow as tf
from MultiGPUTrainer import MultiGPUTrainer
from tensorflow.python.ops import control_flow_ops

from utils import get_variables_to_train, montage_tf

slim = tf.contrib.slim


class AETrainer(MultiGPUTrainer):
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='linear', init_lr=0.0003,
                 end_lr=None, beta1=0.5, num_gpus=1, dataset_id=None):
        MultiGPUTrainer.__init__(self, model, dataset, pre_processor, num_epochs, optimizer, lr_policy, init_lr, end_lr,
                                 beta1, num_gpus, dataset_id)
        self.train_set = self.dataset.get_trainset_unlabelled()
        self.num_train_steps = (self.dataset.get_num_train_unlabelled() / self.model.batch_size) * self.num_epochs

    def make_init_fn(self, chpt_path):
        return None

    def build_model(self, batch_queue, tower, opt, scope):
        imgs_train = batch_queue.dequeue()
        tf.summary.image('images/train', montage_tf(imgs_train, 2, 8), max_outputs=1)

        # Create the model
        dec_im = self.model.net(imgs_train, reuse=True if tower > 0 else None)
        tf.summary.image('images/autoencoder', montage_tf(dec_im, 2, 8), max_outputs=1)

        # Compute losses
        loss = self.model.ae_loss(scope, tower)
        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients for the batch of data on this tower.
        grads = opt.compute_gradients(loss, get_variables_to_train())
        grad_mult = {var.op.name: 2.0 if var.op.name.endswith('biases') else 1.0 for (_, var) in grads}
        print('Gradient multipliers: {}'.format(grad_mult))
        grads = tf.contrib.training.multiply_gradients(grads, grad_mult)
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss, grads, None
