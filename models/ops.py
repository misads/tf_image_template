import tensorflow as tf


def d_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))


def g_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                            kernel_initializer=initializer)


def g_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                      kernel_initializer=initializer)


"""
    ==============================
    convs
"""

"""
    ==============================
    norms
"""


def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
      name: name of the variable
      shape: list of ints
      mean: mean of a Gaussian
      stddev: standard deviation of a Gaussian
    Returns:
      A trainable variable
    """
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant))


def _leaky_relu(input, slope):
    return tf.maximum(slope * input, input)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def _norm(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalization or None
    """
    if norm == 'instance':
        return _instance_norm(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                            decay=0.9,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)


def _instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


"""


"""


def batchnorm(inputs):
    """
        axis=3: feature channel layer
        :param inputs:
        :return:
    """
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def lr_decay(global_step, starter_learning_rate=0.0002, end_learning_rate=0.0, start_decay_step=100000,
             decay_steps=100000):
    """

        :param global_step: a global_step variable tensor
        :param starter_learning_rate:
        :param end_learning_rate:
        :param start_decay_step: after this step, learning_rate will decay
        :param decay_steps: learning_rate will decay to end_learning_rate after decay_steps
            total_step = (start_decay_step + decay_steps)
        :return:
    """
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=1.0),
            starter_learning_rate
        )

    )
    return learning_rate


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)
