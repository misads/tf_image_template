# encoding=utf-8
import os
import numpy as np
import cv2
import tensorflow as tf

from utils.misc_utils import get_file_paths_by_pattern, get_file_name, create_global_step


class PipeLineV1(object):
    def __init__(self, path='train', shuffle=True, reshape=None, batch_size=None, dtype=tf.float32):
        """

            :param path: image folder or image path lists
            :param shuffle:
            :param reshape: be a tuple or a list (width, height)
            :param dtype: convert input images to tf.uint8 (0-255) or tf.float32(0-1)
        """

        if type(path) == str:
            if not os.path.isdir(path):
                raise Exception("Directory '%s' does not exist" % path)
            input_paths = self._get_input_paths(path)
        else:
            input_paths = path

        """
        load images
        """
        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=shuffle)
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = self.__decode(contents)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=dtype)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input.set_shape([None, None, 3])
            if reshape is not None:
                raw_input = tf.image.resize_images(raw_input, size=[reshape[1], reshape[0]],
                                                   method=tf.image.ResizeMethod.BICUBIC)

            print("%d images loaded in '%s'" % (len(input_paths), path))

        self.__images = raw_input
        self.__image_paths = input_paths
        self._size = len(input_paths)
        self._batch_size = batch_size
        self._shuffled = shuffle
        self._image_shape = reshape
        self._paths = paths
        # self._images = []
        # self._labels = []
        # self._step = 0

    def _get_input_paths(self, path):
        input_paths = get_file_paths_by_pattern(path, pattern='*.jpg')
        self.__decode = tf.image.decode_jpeg
        if len(input_paths) == 0:
            input_paths = get_file_paths_by_pattern(path, pattern='*.png')
            self.__decode = tf.image.decode_png

        if len(input_paths) == 0:
            raise Exception("no image files found, only jpg and png supported yet")

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_file_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_file_name(path)))
        else:
            input_paths = sorted(input_paths)
        return input_paths

    def images(self):
        return self.__images

    def steps_per_epoch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        return self._size // batch_size

    def paths(self):
        return self._paths

    def size(self):
        return self._size


PipeLine = PipeLineV1

if __name__ == '__main__':
    a = PipeLine('a', reshape=(280, 280))
    b = PipeLine('b', reshape=(280, 280))

    # imgs = tf.train.batch([a.images()], batch_size=5)
    # imgs = tf.train.batch([b.images()], batch_size=5)
    imgs = tf.train.shuffle_batch(
        [b.images()], batch_size=2, num_threads=1,
        capacity=10,
        min_after_dequeue=3
    )
    incr_global_step = create_global_step()

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        for i in range(20):
            step, _ = sess.run([sv.global_step, incr_global_step])
            print(i, step)

    # train_data.show_some_random_samples(20)
