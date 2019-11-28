# encoding=utf-8
"""
Misc TensorFlow utils

Author: xuhaoyu@tju.edu.cn

update 11.26

Usage:
    `import misc_utils as utils`
    `utils.func_name()`  # to call functions in this file
"""
import glob
import os
import pdb
import random
import sys
import time

import cv2
from PIL import Image
from PIL import ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


#############################
#    TensorFlow  utils
#############################


def im_list_to_tensor_input(ims, height=None, width=None, fixed=True):
    """
        Convert a list of images into a network input
        Assumes images are already prepared (means subtracted, BGR order)

        :param ims: image list
        :param height: optional, input tensor height, if img is not this height, it will be first reshaped to this height
        :param width:
        :param fixed:
            for True: input tensor should be size of [batch, height, width, channel]
            for False: input tensor should be [None ,None, None, 3]

    :return:
    """
    if width and height:
        blobs = []
        for im in ims:
            img = cv2.resize(im, (height, width))
            blobs.append(img)
    else:
        return ims


def read_image_as_tensor(image_path, form='png', cast='float32', channels=0):
    """
        Reads the jpg image from image_path.
        Returns the image as a tf.float32 tensor
        Args:
            image_path: tf.string tensor
            form: 'jpg' or 'png'
            cast: 'uint8' or float32
            channels:
                0: Use the number of channels in the JPEG-encoded image.
                1: output a grayscale image.
                3: output an RGB image.
        Reuturn:
            the decoded jpeg/png image casted to uint8/float32 [height, width, channel]
    """
    if form == 'jpg' or form == 'jpeg':
        decode = tf.image.decode_jpeg
    elif form == 'png':
        decode = tf.image.decode_png
    else:
        raise TypeError('image type not supported')

    dtype = tf.uint8 if 'int' in cast else tf.float32

    # paths, contents = tf.WholeFileReader().read(image_path)
    # return tf.image.convert_image_dtype(
    #     decode(contents, channels=channels),
    #     dtype=dtype)

    return tf.image.convert_image_dtype(
        decode(
            tf.read_file(image_path), channels=channels),
        dtype=dtype)


def set_random_seed(seed=None):
    """
        set the same random seed for random/np.random and tf
        :param seed: an integer, if is None, a random seed will be generated
        :return:
    """
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_global_step():
    """
        get TensorFlow's global_step
        example:
        incr_global_step = create_global_step()
        sv = tf.train.Supervisor()
            with sv.managed_session() as sess:
                for step in range(self._max_steps):
                    step, _ = sess.run([sv.global_step, incr_global_step])
    """
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)
    return incr_global_step


def limit_gpu_usage(max_usage=1.0):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = max_usage
    return tf_config


def allow_gpu_growth_config():
    """
        example:
        tfconfig = allow_gpu_growth_config()
        with tf.Session(config=tfconfig) as sess:
            pass
        :return:
    """
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    return tf_config


def save_ckpt(sess, path='checkpoint/model.ckpt', step=None, max_to_keep=50):
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    saver.save(sess, path, global_step=step)


def restore_ckpt(sess, ckpt_path='checkpoint', latest=True, variables_to_restore=None):
    """
        restore_ckpt
        :param sess:
        :param ckpt_path:
        :param latest:
        :param variables_to_restore: use get_variables_in_ckpt_file(*.ckpt) to get variables to restore
        :return:
    """
    if not variables_to_restore:
        restorer = tf.train.Saver()
    else:
        restorer = tf.train.Saver(variables_to_restore)
    if latest:
        restorer.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    else:
        restorer.restore(sess, ckpt_path)  # .data文件


#  restore graph from .meta file
def restore_graph(sess, graph_path='checkpoint/test.meta', ckpt_path='checkpoint', latest=True):
    """
        restore_graph
        :param sess: should be tf.Session()
        :param graph_path: (*.meta) file
        :param ckpt_path: checkpoint folder or checkpoint folder/checkpoint name (e.g. checkpoint/test)
        :param latest: if restore the latest checkpoint
        :return:
    """
    saver = tf.train.import_meta_graph(graph_path)
    if latest:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    else:
        saver.restore(sess, ckpt_path)  # .data文件


def get_variables_in_ckpt_file(file_name):
    """
        get_variables_in_ckpt_file
        :param file_name : (*.ckpt) file
        :return: var_to_shape_map
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def get_parameter_count():
    """
        Example:
        parameter_count = get_parameter_count()
        print("Parameter Count =", sess.run(parameter_count))

        :return:
    """
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    return parameter_count


def get_all_tf_variables(trainable_only=True, scope=None):
    """
        Example:
        variables = utils.get_all_tf_variables()  # before session
        with tf.Session() as sess:
            for name, shape in sess.run(variables):
                if len(list(shape)) > 1:
                    print("   '%s' shape=%s" % (name, str(shape)))

        :param trainable_only: if True, only return trainable_variables
        :param scope:
        :return:
    """
    if trainable_only:
        variables = tf.trainable_variables()
        variable_shapes = [tf.shape(v) for v in tf.trainable_variables()]
    else:
        variables = tf.global_variables()
        variable_shapes = [tf.shape(v) for v in tf.global_variables()]

    var_list = []
    varshape_list = []
    for k, v in zip(variables, variable_shapes):
        if not scope or k[:len(scope)] == scope:
            var_list.append(tf.constant(k.name))
            varshape_list.append(v)

    return list(zip(var_list, varshape_list))


def print_graph_collections():
    print('Collections')
    print(tf.get_default_graph().get_all_collection_keys())


def print_graph_tensors(scope=None):
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for tensor in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        if not scope or tensor.name[:len(scope)] == scope:
            print("'%s shape=%s'" % (tensor.name, tensor.shape))


def find_tensor_by_name(name, scope=None):
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for tensor in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        if not scope or tensor.name[:len(scope)] == scope:
            if name in tensor.name:
                print("'%s shape=%s'" % (tensor.name, tensor.shape))


def add_to_collection(name, tensor):
    """
        example
        :param name: 'network-output'
        :param tensor: y
        :return:
    """
    tf.add_to_collection(name, tensor)


def get_graph_collection(name=None, index=None):
    if not name:
        return tf.get_default_graph().get_all_collection_keys()

    collection = tf.get_collection(name)
    if index is None:
        return collection
    else:
        return collection[index]


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(name)

