import tensorflow as tf
import os

from globals import LOG_DIR

slim = tf.contrib.slim


class SDNetTester:
    def __init__(self, model, dataset, pre_processor):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.additional_info = None
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.num_eval_steps = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}'.format(self.model.name, self.dataset.name)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def get_test_batch(self, dataset_id=None):
        with tf.device('/cpu:0'):
            if dataset_id:
                test_set = self.dataset.get_split(dataset_id)
                self.num_eval_steps = (self.dataset.get_num_dataset(dataset_id) / self.model.batch_size)
            else:
                test_set = self.dataset.get_testset()
                self.num_eval_steps = (self.dataset.get_num_test() / self.model.batch_size)
            print('Number of evaluation steps: {}'.format(self.num_eval_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False,
                                                                      common_queue_capacity=10 * self.model.batch_size,
                                                                      common_queue_min=self.model.batch_size)

            [img_test, label_test] = provider.get(['image', 'label'])
            label_test -= self.dataset.label_offset

            # Pre-process data
            img_test = self.pre_processor.process_test(img_test)

            # Make batches
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=self.model.batch_size,
                                                    num_threads=1,
                                                    capacity=4*self.model.batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue([imgs_test, labels_test])
            return batch_queue.dequeue()

    def test_classifier(self, num_conv_trained=None, dataset_id=None):
        if not self.additional_info:
            self.additional_info = 'conv_{}'.format(num_conv_trained)
        print('Restoring from: {}'.format(self.get_save_dir()))
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_test, labels_test = self.get_test_batch(dataset_id)

                # Get predictions
                predictions, _ = self.model.classifier(imgs_test, self.dataset.num_classes, training=False)

                # Compute predicted label for accuracy
                preds_test = tf.argmax(predictions, 1)

                # Choose the metrics to compute:
                names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                    'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_test),
                })
                summary_ops = self.make_summaries(names_to_values)

                # Start evaluation
                slim.evaluation.evaluation_loop('', self.get_save_dir(), self.get_save_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=1,
                                                eval_op=names_to_updates.values(),
                                                summary_op=tf.summary.merge(summary_ops))

    def make_summaries(self, names_to_values):
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)
        return summary_ops

    def test_classifier_cv(self, num_conv_trained=None, fold=0):
        self.additional_info = 'conv_{}_fold_{}'.format(num_conv_trained, fold)
        self.test_classifier(num_conv_trained)

