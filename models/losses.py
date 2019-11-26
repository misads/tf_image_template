import tensorflow as tf


def L1(targets, outputs):
    with tf.name_scope("L1_loss"):
        return tf.reduce_mean(tf.abs(targets-outputs))


def L2(targets, outputs):
    with tf.name_scope("L2_loss"):
        return tf.sqrt(tf.nn.l2_loss(tf.abs(targets - outputs)))


def softmax_cross_entropy(targets, outputs):
    """
        example:
        outputs = tf.constant([[4., 5., 3.]])
        targets = tf.constant([[0., 1., 0.]])
        cross_entropy = softmax_cross_entropy(targets, outputs)

        :param targets: should be one-hot encoding
        :param outputs: logits outputs of network
        :return:
    """
    return tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs)


def sparse_softmax_cross_entropy(targets, outputs):
    """
        example:
        outputs = tf.constant([[4., 5., 3.]])
        targets = tf.constant([[1]])  # index 1 is label (one hot=[0, 1, 0])
        cross_entropy = sparse_softmax_cross_entropy(targets, outputs)

        :param targets:
        :param outputs:
        :return:
    """
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)


def d_loss_cross_entropy(predict_real, predict_fake, eps=1e-12):
    """
        predict_real = predicted probability of real to be real, the higher the better
        predict_fake = predicted probability of fake(generated) to be real, the lower the better (for discriminator)

        :param predict_real: D(y)
        :param predict_fake: D(fake_y) or D(G)
        :param eps:
        :return:
    """
    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + eps) + tf.log(1 - predict_fake + eps)))
        return discrim_loss


def g_loss_cross_entropy(predict_fake, eps=1e-12):
    """
        predict_fake = predicted probability of fake(generated) to be real, the higher the better (for generator)
            it means that the generator can generate very **REAL** image
            to cheat the discriminator

        :param predict_fake: D(G)
        :param eps:
        :return:
    """
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_gan = tf.reduce_mean(-tf.log(predict_fake + eps))
        return gen_loss_gan


REAL_LABEL = 0.9


def d_loss_mse(predict_real, predict_fake):

    """
        lsgan
        https://arxiv.org/abs/1611.04076v2

        predict_real = predicted probability of real to be real, the higher the better
        predict_fake = predicted probability of fake(generated) to be real, the lower the better (for discriminator)

        :param predict_real: D(y)
        :param predict_fake: D(fake_y) or D(G)
        :return:
    """
    error_real = tf.reduce_mean(tf.squared_difference(predict_real, REAL_LABEL))
    error_fake = tf.reduce_mean(tf.square(predict_fake))
    with tf.name_scope("discriminator_loss"):
        discrim_loss = (error_real + error_fake) / 2
        return discrim_loss


def g_loss_mse(predict_fake):
    """
        lsgan
        https://arxiv.org/abs/1611.04076v2
        predict_fake = predicted probability of fake(generated) to be real, the higher the better (for generator)
            it means that the generator can generate very **REAL** image
            to cheat the discriminator

        :param predict_fake: D(G)
        :return:
    """
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_gan = tf.reduce_mean(tf.squared_difference(predict_fake, REAL_LABEL))
        return gen_loss_gan
