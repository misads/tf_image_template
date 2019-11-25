# encoding=utf-8
import os
import numpy as np
import cv2


class imdb(object):
    def __init__(self, path='sd', shuffle=True, resize=(-1, -1)):
        self._path = path
        self._shuffle = shuffle
        self._resize = resize
        self._image_sets = []
        self._labels = []
        self._init_data()

    def _init_data(self):
        l = os.listdir(self._path)
        np.random.shuffle(l)
        for f in l:
            path = os.path.join(self._path, f)
            # print(path)
            img = cv2.imread(path)
            if self._resize[0] > 0:
                img_re = cv2.resize(img, self._resize)
                self._image_sets.append(img_re)
                del img_re
            else:
                self._image_sets.append(img)
            del img
        self._dataset_size = len(self._image_sets)
        self._image_sets = np.array(self._image_sets)

    def shuffle_image(self):
        np.random.shuffle(self._image_sets)

    def get_step_nums(self, batch_size=64):
        return self._dataset_size // batch_size

    def get_batch(self, step, batch_size=64):
        assert step > 0, 'step starts from 1.'
        assert step <= self._dataset_size // batch_size, 'dataset EOF'
        i = step - 1
        return self._image_sets[i * batch_size: i * batch_size + batch_size]

    def get_image_at(self, index):
        return self._image_sets[index]

    def show_image_at(self, index):
        cv2.imshow('img', self._image_sets[index])
        cv2.waitKey(0)

    def show_random_image(self):
        index = np.random.randint(0, self._dataset_size)
        cv2.imshow('img', self._image_sets[index])
        cv2.waitKey(0)

    def dataset_size(self):
        return self._dataset_size


if __name__ == '__main__':
    hd = imdb('trainA', resize=(280, 280))
    for i in range(20):
        hd.show_random_image()
