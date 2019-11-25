# encoding=utf-8
import os
import random

import numpy as np
import cv2
import tensorflow as tf

EPS = 1e-12


# CROP_SIZE = 256


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def depart_input_target_pair(raw_input, args):
    # break apart image pair
    width = tf.shape(raw_input)[1]  # [height, width, channels]
    a_images = preprocess(raw_input[:, :width // 2, :])
    b_images = preprocess(raw_input[:, width // 2:, :])

    if args.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    return inputs, targets


def transform(image, args, seed=0, training=True):
    a = args
    r = image
    CROP_SIZE = args.CROP_SIZE

    if training and a.flip:
        r = tf.image.random_flip_left_right(r, seed=seed)

    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    if training:
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
    else:
        r = tf.image.resize_images(r, [CROP_SIZE, CROP_SIZE], method=tf.image.ResizeMethod.AREA)
        return r

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
    if a.scale_size > CROP_SIZE:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    elif a.scale_size < CROP_SIZE:
        raise Exception("scale size cannot be less than crop size")
    return r


def upscale(image, args):
    if args.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [args.CROP_SIZE, int(round(args.CROP_SIZE * args.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


if __name__ == '__main__':
    pass

