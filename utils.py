import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver


def weights_montage(weights, grid_Y, grid_X, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
        weights: tensor of shape [Y, X, NumChannels, NumKernels]
        (grid_Y, grid_X): shape of the grid. Require: NumKernels == grid_Y * grid_X
        pad: number of black pixels around each filter (between them)

    Return:
        Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    x_min = tf.reduce_min(weights, axis=[0, 1, 2])
    x_max = tf.reduce_max(weights, axis=[0, 1, 2])

    weights1 = (weights - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(weights1-1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')+1

    # X and Y dimensions, w.r.t. padding
    Y = weights1.get_shape()[0] + 2 * pad
    X = weights1.get_shape()[1] + 2 * pad

    channels = weights1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def montage_tf(imgs, num_h, num_w):
    """Makes a montage of imgs that can be used in image_summaries.

    Args:
        imgs: Tensor of images
        num_h: Number of images per column
        num_w: Number of images per row

    Returns:
        A montage of num_h*num_w images
    """
    imgs = tf.unstack(imgs)
    img_rows = [None] * num_h
    for r in range(num_h):
        img_rows[r] = tf.concat(axis=1, values=imgs[r * num_w:(r + 1) * num_w])
    montage = tf.concat(axis=0, values=img_rows)
    return tf.expand_dims(montage, 0)


def remove_missing(var_list, model_path):
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
        var_dict = var_list
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:

        if reader.has_tensor(var):
            available_vars[var] = var_dict[var]
        else:
            logging.warning(
                'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
    return var_list


def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
    """Returns a function that assigns specific variables from a checkpoint.

    Args:
        model_path: The full path to the model checkpoint. To get latest checkpoint
          use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
        var_list: A list of `Variable` objects or a dictionary mapping names in the
          checkpoint to the correspoing variables to initialize. If empty or None,
          it would return  no_op(), None.
        ignore_missing_vars: Boolean, if True it would ignore variables missing in
          the checkpoint with a warning instead of failing.
        reshape_variables: Boolean, if True it would automatically reshape variables
          which are of different shape then the ones stored in the checkpoint but
          which have the same number of elements.

    Returns:
        A function that takes a single argument, a `tf.Session`, that applies the
        assignment operation.

    Raises:
        ValueError: If the checkpoint specified at `model_path` is missing one of
                    the variables in `var_list`.
    """
    if ignore_missing_vars:
        var_list = remove_missing(var_list, model_path)

    saver = tf_saver.Saver(var_list, reshape=reshape_variables)

    def callback(session):
        saver.restore(session, model_path)

    return callback


def get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

    print('Variables to train: {}'.format([v.op.name for v in variables_to_train]))

    return variables_to_train


def get_checkpoint_path(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if not ckpt:
        print("No checkpoint in {}".format(checkpoint_dir))
        return None
    return ckpt.model_checkpoint_path
