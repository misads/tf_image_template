import tensorflow as tf


def L1(targets, outputs):
    return tf.reduce_mean(tf.abs(targets-outputs))


def L2(targets, outputs):
    return tf.sqrt(tf.nn.l2_loss(tf.abs(targets - outputs)))


