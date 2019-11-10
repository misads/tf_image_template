# encoding=utf-8
import tensorflow as tf
import argparse

from config import cfg
from models.base_model import BaseModel


def create_hparams():
    return tf.contrib.training.Hparams(
        batch_size=64,
        optimizer='adam',
        lr=0.0001,
        hidden_size=[128, 128, 128],
        img_shape=cfg.IMAGE_SHAPE

    )


def parse_args():
    # 创建一个parser对象
    parser = argparse.ArgumentParser(description='parser demo')

    # str类型的参数(必填)（不需要加-标志）
    parser.add_argument('--dataset', type=str, default='voc_2007')

    # 选择类型的参数
    parser.add_argument('--net', help='Network to use [vgg16 res101]',
                        choices=['vgg16', 'res101'], default='res101')

    # 如果有参数，则执行命令
    parser.add_argument('--resume', '-r', action='store_true', help='restore last checkpoint')

    args = parser.parse_args()

    return args


def train():
    pass


if __name__ == '__main__':
    args = parse_args()

    hparams = create_hparams()
    model = BaseModel(hparams)

    train()
