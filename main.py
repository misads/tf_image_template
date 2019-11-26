# encoding=utf-8
"""
tf_image_template

Author: xuhaoyu@tju.edu.cn

"""
import argparse

from data_loader.pipeline import PipeLine
from models.mynet import MyNet

from utils.misc_utils import *

parser = argparse.ArgumentParser()
# directories
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
# parser.add_argument("--test_dir", help="use --input_dir to assign test_dir")
parser.add_argument("--val_dir", help="path to validation folder containing images")

parser.add_argument("--log_dir", default='logs', help="path to save TensorBoard log files")
parser.add_argument("--output_dir", default='checkpoint', help="where to put output files")

# mode
parser.add_argument("--train", action="store_true", help="train mode")
parser.add_argument("--resume", action="store_true", help="resume training from latest checkpoint in checkpoint/ folder")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--export", action="store_true", help="export mode")

parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

# steps and frequencies
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=500, help="update summaries every summary_freq steps")
parser.add_argument("--eva_freq", type=int, default=5000, help="evaluation every eva_freq steps")
parser.add_argument("--progress_freq", type=int, default=1, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")

# hyper parameters
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
parser.add_argument("--output_filetype", default="png", choices=["png", "jpg"])

args = parser.parse_args()


if __name__ == '__main__':

    set_random_seed(args.seed)
    try_make_dir(args.output_dir)

    print_args(args)  # print training parameters

    if args.train:
        train_inputs = PipeLine(args.input_dir, reshape=None, batch_size=args.batch_size)
        val_inputs = PipeLine(args.val_dir, shuffle=False, reshape=None, batch_size=args.batch_size) if args.val_dir else None

        model = MyNet(args, train_inputs, val_inputs)
        model.train()

    elif args.test:

        test_inputs = PipeLine(args.input_dir, shuffle=False, reshape=None, batch_size=args.batch_size)
        model = MyNet(args, test_inputs)
        model.test()
    else:
        raise Exception('Mode not specified, please use [--train] or [--test]')


