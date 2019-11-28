# encoding=utf-8
"""
A pipeline for tf_image_template

Author: xuhaoyu@tju.edu.cn

update 11.25

"""
import os
import tensorflow as tf

from utils.misc_utils import get_file_paths_by_pattern, get_file_name, color_print
from utils.tf_utils import create_global_step


class PipeLineV1(object):
    def __init__(self, path='train', shuffle=True, reshape=None, batch_size=None, dtype=tf.float32, num_threads=1,
                 min_queue_examples=1000):
        """

            :param path: image folder or image path lists
            :param shuffle:
            :param reshape: be a tuple or a list (width, height)
            :param dtype: convert input images to tf.uint8 (0-255) or tf.float32(0-1)
        """

        """
            check if path is a dir/.tfrecords file or a image file list
        """
        self._tfrecords = False
        if type(path) == str:
            if not os.path.isdir(path):
                if '.tfrecords' in path:
                    self._tfrecords = True
                else:
                    raise Exception("Directory '%s' does not exist" % path)
            else:
                input_paths = self._get_input_paths(path)
        else:
            input_paths = path

        """
        load images
        """
        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=shuffle)
            if self._tfrecords:
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(path_queue)
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'image/file_name': tf.FixedLenFeature([], tf.string),
                        'image/encoded_image': tf.FixedLenFeature([], tf.string),
                    })

                image_buffer = features['image/encoded_image']
                raw_input = tf.image.decode_jpeg(image_buffer, channels=3)

            else:
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

            color_print("%d images loaded in '%s'" % (len(input_paths), path), 2)

        self.__images = raw_input
        self.__image_paths = input_paths
        self._size = len(input_paths)
        self._batch_size = batch_size
        self._shuffled = shuffle
        self._image_shape = reshape
        self._paths = paths
        self._num_threads = num_threads
        self._min_queue_examples = min_queue_examples
        # self._labels = []

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

    def _preprocess(self):
        image = self.__images
        """
            Todo
            Override this function and add your preprocess code here
            
        """
        return image

    def feed(self, batch_size=None):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """

        if batch_size is None:
            batch_size = self._batch_size

        with tf.name_scope(self._paths.replace('.', '_')):
            image = self._preprocess()
            images = tf.train.shuffle_batch(
                [image], batch_size=batch_size, num_threads=self._num_threads,
                capacity=self._min_queue_examples + 3 * batch_size,
                min_after_dequeue=self._min_queue_examples
            )

            tf.summary.image('_input', images)
        return images

    def steps_per_epoch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        return self._size // batch_size

    def paths(self):
        return self._paths

    def size(self):
        return self._size


PipeLine = PipeLineV1


def debug_pipeline():
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


if __name__ == '__main__':
    debug_pipeline()
