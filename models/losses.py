import tensorflow as tf
import models.ops as ops

"""
Commonly used losses
    
Author: xuhaoyu@tju.edu.cn
    
Losses
    :Pixel loss
        :L1
        :L2
    :Cross Entropy loss
        :bin_cross_entropy
        :softmax_cross_entropy
        :sparse_softmax_cross_entropy
    :GAN loss
        :d_loss_cross_entropy
        :g_loss_cross_entropy
        :d_loss_mse
        :g_loss_mse
    :Perceptual loss
        :perceptual_similarity_loss
        
Parameters
    :targets gt labels, be size of [batch, height, width, channel]
    :outputs
"""


#######################
#    Pixel loss
#######################

def L1(targets, outputs):
    with tf.name_scope("L1_loss"):
        return tf.reduce_mean(tf.abs(targets-outputs))


def L2(targets, outputs):
    with tf.name_scope("L2_loss"):
        # l2 = tf.square(targets - outputs) / 2
        # return tf.reduce_mean(tf.sqrt(l2))
        return tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(tf.abs(targets - outputs))))


#######################
#  Cross Entropy loss
#######################

def bin_cross_entropy(targets, outputs):
    """
        Example:
            targets: [1., 0., 0., 1.]
            outputs: [0.9, 0.4, 0.2, 0.4]
            return: 0.4389051
        :param targets:
        :param outputs:
        :return:
    """
    z = targets
    x = outputs
    log = ops.safe_log
    cross_entropy = - z * log(x) - (1-z) * log(1-x)
    return tf.reduce_mean(cross_entropy)


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


#######################
#       GAN loss
#######################

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


#######################
#   Perceptual loss
#######################

def perceptual_similarity_loss(targets, outputs, vgg, layers=('pool2', 'pool5')):
    """
        VGG 16 perceptual loss
        :param targets:
        :param outputs:
        :param vgg: vgg model (defined in vgg16.py)
        :param layers: ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5']
        :return:
    """

    targets_224 = tf.image.resize_images(targets, [224, 224])  # to feed vgg, need resize
    outputs_224 = tf.image.resize_images(outputs, [224, 224])

    target_features = vgg.build(targets_224, layers)
    outputs_features = vgg.build(outputs_224, layers)

    sum = None
    for f1, f2 in zip(target_features, outputs_features):
        if sum is None:
            sum = tf.reduce_mean(tf.squared_difference(f1, f2))
        else:
            sum = sum + tf.reduce_mean(tf.squared_difference(f1, f2))

    return sum / len(layers) * 0.00001