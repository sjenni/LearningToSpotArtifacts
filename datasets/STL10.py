import os
import tensorflow as tf
from globals import STL10_TF_DATADIR

slim = tf.contrib.slim


class STL10:

    SPLITS_TO_SIZES = {'train_unlabeled': 100000, 'train': 5000, 'test': 8000, 'train_fold_0': 4000,
                       'train_fold_1': 4000,
                       'train_fold_2': 4000, 'train_fold_3': 4000, 'train_fold_4': 4000, 'train_fold_5': 4000,
                       'train_fold_6': 4000, 'train_fold_7': 4000, 'train_fold_8': 4000, 'train_fold_9': 4000,
                       'test_fold_0': 1000, 'test_fold_1': 1000, 'test_fold_2': 1000, 'test_fold_3': 1000,
                       'test_fold_4': 1000, 'test_fold_5': 1000, 'test_fold_6': 1000, 'test_fold_7': 1000,
                       'test_fold_8': 1000, 'test_fold_9': 1000
                       }

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [96 x 96 x 3] color image.',
        'label': 'A single integer between 0 and 9 or -1 for unlabeled',
        'cartoon': 'A [96 x 96 x 3] cartooned image.',
        'edges': 'A [96 x 96 x 1] edge map.'
    }

    def __init__(self):
        self.reader = tf.TFRecordReader
        self.label_offset = 0
        self.is_multilabel = False
        self.data_dir = STL10_TF_DATADIR
        self.file_pattern = 'stl10_%s.tfrecord'
        self.num_classes = 10
        self.name = 'STL10'

    def get_keys_to_features(self):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature((), tf.int64, default_value=96),
            'image/width': tf.FixedLenFeature((), tf.int64, default_value=96),
            'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=-1),
        }
        return keys_to_features

    def get_items_to_handlers(self):
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', shape=[96, 96, 3], channels=3),
            'height': slim.tfexample_decoder.Tensor('image/height'),
            'width': slim.tfexample_decoder.Tensor('image/width'),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }
        return items_to_handlers

    def get_trainset_labelled(self):
        return self.get_split('train')

    def get_trainset_unlabelled(self):
        return self.get_split('train_unlabeled')

    def get_testset(self):
        return self.get_split('test')

    def get_train_fold_id(self, fold_idx):
        return 'train_fold_{}'.format(fold_idx)

    def get_test_fold_id(self, fold_idx):
        return 'test_fold_{}'.format(fold_idx)

    def get_num_train_labelled(self):
        return self.SPLITS_TO_SIZES['train']

    def get_num_train_unlabelled(self):
        return self.SPLITS_TO_SIZES['train_unlabeled']

    def get_num_test(self):
        return self.SPLITS_TO_SIZES['test']

    def get_split_size(self, split_name):
        return self.SPLITS_TO_SIZES[split_name]

    def format_labels(self, labels):
        return slim.one_hot_encoding(labels, self.num_classes)

    def get_split(self, split_name, data_dir=None):
        """Gets a dataset tuple with instructions for reading ImageNet.
        Args:
          split_name: A train/eval split name.
          data_dir: The base directory of the dataset sources.
        Returns:
          A `Dataset` namedtuple.
        Raises:
          ValueError: if `split_name` is not a valid train/eval split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        if not data_dir:
            data_dir = self.data_dir

        tf_record_pattern = os.path.join(data_dir, self.file_pattern % split_name)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset at %s' % data_dir)

        # Build the decoder
        keys_to_features = self.get_keys_to_features()
        items_to_handlers = self.get_items_to_handlers()
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
            data_sources=data_files,
            reader=self.reader,
            decoder=decoder,
            num_samples=self.SPLITS_TO_SIZES[split_name],
            items_to_descriptions=self.ITEMS_TO_DESCRIPTIONS
        )
