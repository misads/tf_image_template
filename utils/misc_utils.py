# encoding=utf-8
"""
Misc system, image process and TensorFlow utils

Author: xuhaoyu@tju.edu.cn

update 11.25

"""
import glob
import os
import pdb
import random
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import matplotlib.pyplot as plt


#############################
#    System utils
#############################


def p(v):
    """
        recursively print
        :param v:
        :return:
    """
    if type(v) == list or type(v) == tuple:
        for i in v:
            print(i)
    elif type(v) == dict:
        for k in v:
            print('%s: %s' % (k, v[k]))
    else:
        print(v)


def color_print(text='', color=0):
    """
        :param text:
        :param color:
            0       black
            1       red
            2       green
            3       yellow
            4       blue
            5       cyan (like light red)
            6       magenta (like light blue)
            7       white

        :return:
    """
    print('\033[1;3%dm' % color, end='')
    print(text, end='')
    print('\033[0m')


def print_args(args):
    """
        example
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            print_args(args)

        :param args: args parsed by argparse
        :return:
    """
    for k, v in args._get_kwargs():
        print('\033[1;32m', k, "\033[0m=\033[1;33m", v, '\033[0m')


def safe_key(dic, key, default=None):
    """
        return dict[key] only if dict has this key
        in case of KeyError
        :param dic:
        :param key:
        :param default:
        :return:
    """
    if key in dic:
        return dic[key]
    else:
        return default


def try_make_dir(folder):
    """
        in case of FileExistsError
        :param folder:
        :return:
    """
    os.makedirs(folder, exist_ok=True)


def get_file_name(path):
    """
        example:
            get_file_name('train/0001.jpg')
            returns 0001

        :param path:
        :return: filename
    """
    name, _ = os.path.splitext(os.path.basename(path))
    return name


def get_file_paths_by_pattern(folder, pattern='*'):
    """
        examples: get all *.png files in folder
            get_file_paths_by_pattern(folder, '*.png')
        get all files with '_rotate' in name
            get_file_paths_by_pattern(folder, '*rotate*')

        :param folder:
        :param pattern:
    :return:
    """
    paths = glob.glob(os.path.join(folder, pattern))
    return paths


def format_time(seconds):
    """
        examples:
            format_time(10) -> 10s
            format_time(100) -> 1m
            format_time(10000) -> 2h 47m
            format_time(1000000) -> 11d 13h 47m
        :param seconds:
        :return:
    """
    eta_d = seconds // 86400
    eta_h = (seconds % 86400) // 3600
    eta_m = (seconds % 3600) // 60
    eta_s = seconds % 60
    if eta_d:
        eta = '%dd %dh %dm' % (eta_d, eta_h, eta_m)
    elif eta_h:
        eta = '%dh %dm' % (eta_h, eta_m)
    elif eta_m:
        eta = '%dm' % eta_m
    else:
        eta = '%ds' % eta_s
    return eta


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 80


TOTAL_BAR_LENGTH = 30
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, pre_msg=None, msg=None):
    """

        :param current: from 0 to total-1
        :param total:
        :param pre_msg: msg **before** progress bar
        :param msg: msg **after** progress bar
        :return:
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    if pre_msg is None:
        pre_msg = ''
    sys.stdout.write(pre_msg + ' Step:')

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    eta_time = int((total - current) * step_time)
    eta = format_time(eta_time)

    L = []
    L.append(' ETA: %s' % eta)
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(3):
        sys.stdout.write(' ')
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    # sys.stdout.write(' %d/%d ' % (current+1, total))
    for i in range(len(msg) + int(TOTAL_BAR_LENGTH/2)+8):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


#############################
#    Image process utils
#############################


def histogram_demo(image, title=None):
    if title:
        plt.title(title)

    plt.hist(image.ravel(), 256, [0, 256])  # 直方图
    plt.show()


def hwc_to_whc(img):
    """
        change [height][width][channel] to [width][height][channel]
        :param img:
        :return:
    """
    return img
    img = np.transpose(img, [1, 0, 2])
    return img


def is_file_image(filename):
    img_ex = ['jpg', 'png', 'bmp', 'jpeg', 'tiff']
    if '.' not in filename:
        return False
    s = filename.split('.')

    if s[-1].lower() not in img_ex:
        return False

    return True


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


def allow_gpu_growth_config():
    """
        example:
        tfconfig = allow_gpu_growth_config()
        with tf.Session(config=tfconfig) as sess:
            pass
        :return:
    """
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True


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


def print_param_count(sess):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    print("Parameter Count =", sess.run(parameter_count))


def print_graph_variables(trainable_only=False, sess=None, scope=None):
    if trainable_only:
        variables = [v for v in tf.trainable_variables()]
        variable_shapes = [tf.shape(v) for v in tf.trainable_variables()]
    else:
        variables = [v for v in tf.global_variables()]
        variable_shapes = [tf.shape(v) for v in tf.global_variables()]

    if trainable_only:
        print('Trainable ', end='')

    print('Variables')

    if sess is not None:
        # sess.run(tf.shape(variables)) to get exact shape of variables
        values = sess.run(variable_shapes)
        for k, v in zip(variables, values):
            if not scope or k[:len(scope)] == scope:
                print("   '%s' shape=%s" % (k.name, v))
    else:
        for n in variables:
            if not scope or n[:len(scope)] == scope:
                print("   '%s' shape=%s" % (n.name, n.shape))


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


#############################
#    TensorFlow  functions & layers
#############################


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
