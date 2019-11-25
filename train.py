# encoding=utf-8
import random

import numpy as np
import tensorflow as tf
import argparse

from config import cfg
from data_loader.pipeline import PipeLine
from models.mynet import MyNet

from utils.misc_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--train", action="store_true", help="train mode")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--export", action="store_true", help="export mode")
parser.add_argument("--eva_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=1000, help="update summaries every summary_freq steps")
parser.add_argument("--eva_freq", type=int, default=6995, help="evaluation every eva_freq steps")
parser.add_argument("--progress_freq", type=int, default=1, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_size", dest='CROP_SIZE', type=int, default=256, help="cropping to this size")


parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

args = parser.parse_args()


if __name__ == '__main__':

    set_random_seed(args.seed)
    try_make_dir(args.output_dir)

    print_args(args)  # print training parameters

    train_inputs = PipeLine(args.input_dir, reshape=None, batch_size=args.batch_size)
    eva_inputs = PipeLine(args.eva_dir, shuffle=False, reshape=None, batch_size=args.batch_size)
    model = MyNet(args, train_inputs, eva_inputs, training=True)

    model.train()
