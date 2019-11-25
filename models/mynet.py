# encoding=utf-8
import math
import time

import tensorflow as tf

from data_loader.transform import transform, convert
from models.base_model import BaseModel
from models.losses import create_generator, create_discriminator
from utils.misc_utils import *

class MyNet(BaseModel):
    def __init__(self, args, train_inputs, eva_inputs, training=True):
        self._train_inputs = train_inputs
        self._eva_inputs = eva_inputs
        self._images = {}
        self._eva = {}
        BaseModel.__init__(self, args, training=training)

        # alter by self.set_input

    def _preprocess(self, image):
        pass

    def _deprocess(self, image):
        pass

    def _build_graph(self):
        self._manage_inputs()

        self._create_model()

        self._manage_outputs()  # summary and evaluation

    def _manage_outputs(self):
        """
            Evaluation, summary and visualization
        :return:
        """
        # inputs = self._deprocess(self._batch['inputs'])
        # targets = self._deprocess(self._batch['targets'])
        # outputs = self._deprocess(self._outputs['outputs'])
        #
        # self._images['outputs'] = outputs
        #
        # args = self._args
        # with tf.name_scope("convert_inputs"):
        #     converted_inputs = convert(inputs, args)
        #
        # with tf.name_scope("convert_targets"):
        #     converted_targets = convert(targets, args)
        #
        # with tf.name_scope("convert_outputs"):
        #     converted_outputs = convert(outputs, args)
        #
        # self._tensors['ssim'] = tf.image.ssim(self._images['targets'], self._images['outputs'], max_val=1)
        # self._tensors['psnr'] = tf.image.psnr(self._images['targets'], self._images['outputs'], max_val=1)
        #
        # # run_display = {
        # #     'paths': self._batch['paths'],
        # #     'inputs': tf.map_fn(tf.image.encode_png, inputs, dtype=tf.string, name='input_pngs'),
        # #     'targets': tf.map_fn(tf.image.encode_png, targets, dtype=tf.string, name='target_pngs'),
        # #     'outputs': tf.map_fn(tf.image.encode_png, outputs, dtype=tf.string, name='output_pngs'),
        # # }
        # with tf.name_scope('inputs_summary'):
        #     tf.summary.image('inputs', converted_inputs)
        #
        # with tf.name_scope('targets_summary'):
        #     tf.summary.image('targets', converted_targets)
        #
        # with tf.name_scope('outputs_summary'):
        #     tf.summary.image('outputs', converted_outputs)

    def _manage_inputs(self):
        #
        # args = self._args
        # train_inputs = self._train_inputs.images()
        # eva_inputs = self._eva_inputs.images()
        #
        # paths = self._train_inputs.paths()
        # val_paths = self._eva_inputs.paths()
        #
        # inputs, targets = self._depart_input_target_pair(train_inputs)
        #
        # with tf.name_scope('evaluation'):
        #     eval_inputs, eval_targets = self._depart_input_target_pair(eva_inputs)
        #
        #     with tf.name_scope('evainputs'):
        #         eval_input_images = tf.image.resize_images(eval_inputs, [args.CROP_SIZE, args.CROP_SIZE],
        #                                                    method=tf.image.ResizeMethod.AREA)
        #
        #     with tf.name_scope('gts'):
        #         eval_gt_images = tf.image.resize_images(eval_targets, [args.CROP_SIZE, args.CROP_SIZE],
        #                                                 method=tf.image.ResizeMethod.AREA)
        #
        # seed = random.randint(0, 2 ** 31 - 1)
        # with tf.name_scope("input_images"):
        #     input_images = transform(inputs, args, seed)
        #
        # with tf.name_scope("target_images"):
        #     target_images = transform(targets, args, seed)
        #
        # with tf.name_scope("input_batch"):
        #     self._batch['paths'], self._batch['inputs'], self._batch['targets'] = tf.train.batch(
        #         [paths, input_images, target_images],
        #         batch_size=args.batch_size)
        #
        # with tf.name_scope("val_batch"):
        #     self._val['paths'], self._val['inputs'], self._val['targets'] = tf.train.batch(
        #         [val_paths, eval_input_images, eval_gt_images],
        #         batch_size=args.batch_size)
        #
        #     self._images['inputs'] = self._deprocess(self._val['inputs'])
        #     self._images['targets'] = self._deprocess(self._val['targets'])
        #
        # self._steps_per_epoch = self._train_inputs.steps_per_epoch()

    def _create_model(self):

        self._training = tf.placeholder(tf.bool)

        def f1(): return [self._batch['paths'], self._batch['inputs'], self._batch['targets']]

        def f2(): return [self._val['paths'], self._val['inputs'], self._val['targets']]

        paths, inputs, targets = tf.cond(self._training, f1, f2)

        self._debug['paths'] = paths

        args = self._args
        # inputs = self._batch['inputs']
        # targets = self._batch['targets']

        EPS = 1e-12

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = create_generator(inputs, out_channels, args.ngf)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets, args.ndf)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):  # 这里使用reuse=True共享参数
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs, args.ndf)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * args.gan_weight + gen_loss_L1 * args.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        self._model['predict_real'] = predict_real
        self._model['predict_fake'] = predict_fake
        self._losses['discrim_loss'] = ema.average(discrim_loss)
        self._model['discrim_grads_and_vars'] = discrim_grads_and_vars
        self._losses['gen_loss_GAN'] = ema.average(gen_loss_GAN)
        self._losses['gen_loss_L1'] = ema.average(gen_loss_L1)
        self._model['gen_grads_and_vars'] = gen_grads_and_vars
        self._outputs['outputs'] = outputs
        self._train = tf.group(update_losses, incr_global_step, gen_train)


    def train(self):
        args = self._args

        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        logdir = args.output_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None

        tfconfig = allow_gpu_growth_config()
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=tfconfig) as sess:
            # print parameter info
            print("Parameter Count =", sess.run(parameter_count))

            # for k, v in sess.run(variables):
            #     print("   '%s' shape=%s" % (k, str(v)))

            if args.max_epochs is not None:
                self._max_steps = self._steps_per_epoch * args.max_epochs
            if args.max_steps is not None:
                self._max_steps = args.max_steps

            start = time.time()

            for step in range(self._max_steps):
                self._step = step
                run_dict = {
                    "train": self._train,
                    "global_step": sv.global_step,
                }
                if self._is_time_to(args.progress_freq):
                    run_dict.update(self._losses)
                    run_dict['debug'] = self._debug

                if self._is_time_to(args.summary_freq):
                    run_dict['summary'] = sv.summary_op

                results = sess.run(run_dict, feed_dict={self._training: True})

                if self._is_time_to(args.summary_freq):
                    sv.summary_writer.add_summary(results['summary'], results['global_step'])

                if self._is_time_to(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / self._steps_per_epoch)
                    train_step = (results["global_step"] - 1) % self._steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (self._max_steps - step) * args.batch_size / rate

                    # print(results['debug'])
                    msg = '(loss) d:%.3f g:%.3f L1:%.3f | ETA: %s' % (
                        results["discrim_loss"], results["gen_loss_GAN"], results["gen_loss_L1"],
                        format_time(remaining))
                    pre_msg = 'Epoch:%d ' % train_epoch
                    if train_epoch > 0 and train_step > 0:
                        progress_bar(train_step - 1, self._steps_per_epoch, pre_msg, msg)

                    # print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    #     train_epoch, train_step, rate, remaining / 60))
                    # print("discrim_loss", results["discrim_loss"])
                    # print("gen_loss_GAN", results["gen_loss_GAN"])
                    # print("gen_loss_L1", results["gen_loss_L1"])

                """
                    Evaluation
                """
                if step != 0 and step % args.eva_freq == 0:  # 6995
                    self._eva['ssim'] = []
                    self._eva['psnr'] = []
                    print()
                    sys.stdout.flush()
                    for i in range(500//args.batch_size):
                        progress_bar(i, 500//args.batch_size, 'Eva.... ')
                        run_dict = self._tensors
                        run_dict['debug'] = self._debug
                        run_dict['images'] = self._images
                        # run_dict = self._images
                        results = sess.run(run_dict, feed_dict={self._training: False})
                        self.eval(results)

                    print('psnr: %f' % np.average(self._eva['psnr']))
                    print('ssim: %f' % np.average(self._eva['ssim']))

    def _eval(self, results):

        # ssim = tf.image.ssim(results['outputs'], results['targets'], max_val=1.0)
        #print()
        outputs = results['images']['outputs']
        targets = results['images']['targets']
        inputs = results['images']['inputs']

        #print(results['debug']['paths'])

        # outputs = cv2.cvtColor(outputs[0], cv2.COLOR_BGR2RGB)
        # #targets = cv2.cvtColor(targets[0], cv2.COLOR_BGR2RGB)
        # inputs = cv2.cvtColor(inputs[0], cv2.COLOR_BGR2RGB)

        # cv2.imshow('inputs', inputs[0])
        # cv2.imshow('targets', targets[0])
        # cv2.imshow('outputs', outputs[0])
        # cv2.waitKey(0)
        self._eva['psnr'].append(np.average(results['psnr']))
        self._eva['ssim'].append(np.average(results['ssim']))



        #sys.stdout.flush()
