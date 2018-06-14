import os
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf

import dataset_utils
from globals import STL10_DATADIR, STL10_TF_DATADIR

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = os.path.join(STL10_DATADIR, 'stl10_binary/train_X.bin')

# path to the binary train file with labels
LABEL_PATH = os.path.join(STL10_DATADIR, 'stl10_binary/train_y.bin')


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def _add_to_tfrecord(data_filename, tfrecord_writer, label_filename=None, subset=None, test_fold=False):
    """ Loads data from the stl10 files and writes to a TFRecord.
    Args:
      data_filename: The filename of the stl10 file.
      tfrecord_writer: The TFRecord writer to use for writing.
    Returns:
      The new offset.
    """

    images = read_all_images(data_filename)
    labels = None
    if label_filename:
        labels = read_labels(label_filename)

    if subset:
        if test_fold:
            images = images[subset]
            labels = labels[subset]
        else:
            images = np.delete(images, subset, axis=0)
            labels = np.delete(labels, subset, axis=0)

    num_images = images.shape[0]

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()

        with tf.Session(''):
            for j in range(num_images):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    data_filename, j + 1, num_images))
                sys.stdout.flush()

                # Get image, edge-map and cartooned image
                image = np.squeeze(images[j])
                if label_filename:
                    label = labels[j] - 1  # labels should be 0 indexed!
                else:
                    label = -1

                # Encode the images
                image_str = coder.encode_jpeg(image)

                # Build example
                example = dataset_utils.image_to_tfexample(image_str, 'jpg', HEIGHT, WIDTH, int(label))
                tfrecord_writer.write(example.SerializeToString())

    return num_images


def download_and_extract():
    """
        Downloads the stl-10 dataset
    """
    dest_directory = STL10_DATADIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\rDownloading %s %.2f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def run():
    # download data if needed
    download_and_extract()

    # check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print('Found images: {}'.format(images.shape))
    labels = read_labels(LABEL_PATH)
    print('Found labels: {}'.format(labels.shape))

    # Make dataset folder if necessary
    if not os.path.exists(STL10_TF_DATADIR):
        os.mkdir(STL10_TF_DATADIR)

    # Make unlabeled dataset
    unlabeled_train_filename = os.path.join(STL10_DATADIR, 'stl10_binary/unlabeled_X.bin')
    unlabeled_train_tf_file = os.path.join(STL10_TF_DATADIR, 'stl10_train_unlabeled.tfrecord')
    with tf.python_io.TFRecordWriter(unlabeled_train_tf_file) as tfrecord_writer:
        num_written = _add_to_tfrecord(unlabeled_train_filename, tfrecord_writer)
    print('Wrote {} images to {}'.format(num_written, unlabeled_train_tf_file))

    # Make record with the full clean dataset
    labeled_train_filename = os.path.join(STL10_DATADIR, 'stl10_binary/train_X.bin')
    labeled_train_tf_file = os.path.join(STL10_TF_DATADIR, 'stl10_train.tfrecord')
    label_file_train = os.path.join(STL10_DATADIR, 'stl10_binary/train_y.bin')
    with tf.python_io.TFRecordWriter(labeled_train_tf_file) as tfrecord_writer:
        num_written = _add_to_tfrecord(labeled_train_filename, tfrecord_writer, label_filename=label_file_train)
    print('Wrote {} images to {}'.format(num_written, labeled_train_tf_file))

    # Same for clean test
    labeled_test_filename = os.path.join(STL10_DATADIR, 'stl10_binary/test_X.bin')
    labeled_test_tf_file = os.path.join(STL10_TF_DATADIR, 'stl10_test.tfrecord')
    label_file_test = os.path.join(STL10_DATADIR, 'stl10_binary/test_y.bin')
    with tf.python_io.TFRecordWriter(labeled_test_tf_file) as tfrecord_writer:
        num_written = _add_to_tfrecord(labeled_test_filename, tfrecord_writer, label_filename=label_file_test)
    print('Wrote {} images to {}'.format(num_written, labeled_test_tf_file))


if __name__ == '__main__':
    run()
