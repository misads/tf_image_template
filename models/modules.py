import tensorflow as tf

from models.ops import g_deconv, batchnorm, g_conv, d_conv
from models.ops import lrelu


class Generator(object):
    def __init__(self, generator_outputs_channels, first_conv_filters=64):
        self._generator_outputs_channels = generator_outputs_channels
        self._first_conv_filters = first_conv_filters

    def __call__(self, inputs):
        first_conv_filters = self._first_conv_filters
        generator_outputs_channels = self._generator_outputs_channels

        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = g_conv(inputs, first_conv_filters)
            layers.append(output)

        layer_specs = [
            first_conv_filters * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            first_conv_filters * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            first_conv_filters * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            first_conv_filters * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            first_conv_filters * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            first_conv_filters * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            first_conv_filters * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = g_conv(rectified, out_channels)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (first_conv_filters * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (first_conv_filters * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (first_conv_filters * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (first_conv_filters * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (first_conv_filters * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (first_conv_filters * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (first_conv_filters, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = g_deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = g_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return output

#
# def generator(generator_inputs, generator_outputs_channels, first_conv_filters=64):
#     layers = []
#
#     # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
#     with tf.variable_scope("encoder_1"):
#         output = g_conv(generator_inputs, first_conv_filters)
#         layers.append(output)
#
#     layer_specs = [
#         first_conv_filters * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
#         first_conv_filters * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
#         first_conv_filters * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
#         first_conv_filters * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
#         first_conv_filters * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
#         first_conv_filters * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
#         first_conv_filters * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
#     ]
#
#     for out_channels in layer_specs:
#         with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
#             rectified = lrelu(layers[-1], 0.2)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
#             convolved = g_conv(rectified, out_channels)
#             output = batchnorm(convolved)
#             layers.append(output)
#
#     layer_specs = [
#         (first_conv_filters * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
#         (first_conv_filters * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
#         (first_conv_filters * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
#         (first_conv_filters * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
#         (first_conv_filters * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
#         (first_conv_filters * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
#         (first_conv_filters, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
#     ]
#
#     num_encoder_layers = len(layers)
#     for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
#         skip_layer = num_encoder_layers - decoder_layer - 1
#         with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
#             if decoder_layer == 0:
#                 # first decoder layer doesn't have skip connections
#                 # since it is directly connected to the skip_layer
#                 input = layers[-1]
#             else:
#                 input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
#
#             rectified = tf.nn.relu(input)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
#             output = g_deconv(rectified, out_channels)
#             output = batchnorm(output)
#
#             if dropout > 0.0:
#                 output = tf.nn.dropout(output, keep_prob=1 - dropout)
#
#             layers.append(output)
#
#     # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
#     with tf.variable_scope("decoder_1"):
#         input = tf.concat([layers[-1], layers[0]], axis=3)
#         rectified = tf.nn.relu(input)
#         output = g_deconv(rectified, generator_outputs_channels)
#         output = tf.tanh(output)
#         layers.append(output)
#
#     return output

