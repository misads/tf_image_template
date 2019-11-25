# encoding=utf-8
from config import cfg
from utils.misc_utils import *
from models.mynet import MyNet
import tensorflow as tf


def create_hparams():
    return tf.contrib.training.HParams(
        batch_size=64,
        optimizer='adam',
        learning_rate=0.0001,
        hidden_size=[128, 128, 128],
        img_shape=cfg.IMAGE_SHAPE

    )


def test():
    hparams = create_hparams()
    #mynet = MyNet(hparams)
    #tf.data.Dataset.map(tf.read_file())

    print(get_file_paths_by_pattern('train', '*.png'))

    filename_queue = tf.train.string_input_producer(['1.png', '2.png'], shuffle=False)
    x = read_image_as_tensor(filename_queue, form='png', cast='float')
    image = tf.image.resize_images(x, size=(256, 256))
    image.set_shape([256, 256, 3])

    f = tf.layers.flatten(image)
    d = tf.layers.dense(f, 1)

    imgs = tf.train.batch([image], batch_size=2)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        #sess.run(tf.global_variables_initializer())

        #print_graph_variables(trainable_only=False)
        #print_param_count(sess)

        # images = tf.train.shuffle_batch(
        #     [image], batch_size=1, num_threads=2,
        #     capacity=5000,
        #     min_after_dequeue=1000
        # )
        #tf.local_variables_initializer().run()

        #threads = tf.train.start_queue_runners(sess=sess)

        for i in range(20):
            img = sess.run(imgs)
            print(i, img.shape)


if __name__ == '__main__':
    test()
