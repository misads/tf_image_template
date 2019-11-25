# encoding=utf-8
import os
import numpy as np
import cv2

from utils.misc_utils import progress_bar


class DataLoader(object):
    def __init__(self, path='train', shuffle=True, reshape=None):
        """

            :param path:
            :param shuffle:
            :param reshape: be a tuple or a list (width, height)
        """
        self._data_root = path
        self._shuffle = shuffle
        self._image_shape = reshape
        self._images = []
        self._labels = []
        self._step = 0

        self._load_data()

    def _load_data(self):
        l = os.listdir(self._data_root)
        l.sort()
        if self._shuffle:
            np.random.shuffle(l)
        length = len(l)
        for f in l:
            progress_bar(f, length, 'load images... ')
            path = os.path.join(self._data_root, f)
            img = cv2.imread(path)
            if self._image_shape is not None:
                img_re = cv2.resize(img, self._image_shape)
                self._images.append(img_re)
                del img_re
            else:
                self._images.append(img)
            del img
        size = len(self._images)
        print("%d images loaded in '%s'" % (size, self._data_root))
        self._size = size
        self._images = np.array(self._images)

    def shuffle(self):
        np.random.shuffle(self._images)

    def steps_per_epoch(self, batch_size=64):
        return self._size // batch_size

    def get_next_batch(self, batch_size=64):
        i = self._step
        self._step = self._step + 1
        if self._step > self._size // batch_size:
            self._step = 0
        return self._images[i * batch_size: i * batch_size + batch_size]

    def get_random_batch(self, batch_size=64):
        indices = np.random.choice(np.arange(self.size()), batch_size, replace=False)
        return self._images[indices]

    def get_image_at(self, index):
        return self._images[index]

    def show_image_at(self, index):
        cv2.imshow('img', self._images[index])
        cv2.waitKey(0)

    def show_some_random_samples(self, nums=1):
        for i in range(nums):
            index = np.random.randint(0, self._size)
            cv2.imshow('img', self._images[index])
            cv2.waitKey(0)

    def size(self):
        return self._size


if __name__ == '__main__':
    train_data = DataLoader('trainA', reshape=(280, 280))

    train_data.show_some_random_samples(20)

