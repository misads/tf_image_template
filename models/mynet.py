# encoding=utf-8
"""
A network example for tf_image_template

Author: xuhaoyu@tju.edu.cn

update 11.25

"""
import math

from models.losses import L1, L2
from models.process import transform, depart_input_target_pair, deprocess, upscale
from models.base_model import BaseModel
from models.module import generator
from utils.misc_utils import *


class MyNet(BaseModel):
    def __init__(self, args, *pipelines):
        BaseModel.__init__(self, args)

        if args.train:
            self._pipelines['train'] = pipelines[0]
            if args.val_dir:
                self._pipelines['val'] = pipelines[1]

        elif args.test:
            self._pipelines['val'] = pipelines[0]

        # self._train_inputs = train_inputs
        # self._eva_inputs = val_inputs
        self._build_graph()

    def _manage_outputs(self):
        """
            Evaluation, summary and visualization
        :return:
        """
        inputs = deprocess(self._batch['inputs'])
        targets = deprocess(self._batch['targets'])
        outputs = deprocess(self._outputs['outputs'])
        self._images['outputs'] = outputs

        args = self._args
        """
            ======================================
            Todo:
                Add your output converts here
            ======================================
        """
        with tf.name_scope("convert_inputs"):
            converted_inputs = upscale(inputs, args)

        with tf.name_scope("convert_targets"):
            converted_targets = upscale(targets, args)

        with tf.name_scope("convert_outputs"):
            converted_outputs = upscale(outputs, args)
            self._images['converted_outputs'] = converted_outputs


        """
            ======================================
            Todo:
                Add your Evaluation ops here
            ======================================
        """
        if self._validation:
            self._eva['ssim'] = tf.image.ssim(self._images['targets'], self._images['outputs'], max_val=1)
            self._eva['psnr'] = tf.image.psnr(self._images['targets'], self._images['outputs'], max_val=1)




        """
            ======================================
            Todo:
                Add your summaries here
            ======================================
        """
        with tf.name_scope('inputs_summary'):
            tf.summary.image('inputs', converted_inputs)

        with tf.name_scope('targets_summary'):
            tf.summary.image('targets', converted_targets)

        with tf.name_scope('outputs_summary'):
            tf.summary.image('outputs', converted_outputs)

    def _manage_inputs(self):

        args = self._args

        """
            Manage training inputs
        """
        if args.train:
            with tf.name_scope('training'):
                train_images = self._pipelines['train'].images()
                train_paths = self._pipelines['train'].paths()

                seed = random.randint(0, 2 ** 31 - 1)
                train_inputs, train_targets = depart_input_target_pair(train_images, args)
                with tf.name_scope("train_inputs"):
                    train_input_images = transform(train_inputs, args, seed, training=True)

                with tf.name_scope("train_targets"):
                    train_target_images = transform(train_targets, args, seed, training=True)

                with tf.name_scope("train_batch"):
                    self._train_batch['paths'], self._train_batch['inputs'], self._train_batch['targets'] = tf.train.batch(
                        [train_paths, train_input_images, train_target_images],
                        batch_size=args.batch_size)

            self._steps_per_epoch = self._pipelines['train'].steps_per_epoch()

        """
            Manage test or validation (when training)
        """
        if self._validation:
            with tf.name_scope('validation'):
                val_images = self._pipelines['val'].images()
                val_paths = self._pipelines['val'].paths()

                val_inputs, val_targets = depart_input_target_pair(val_images, args)

                with tf.name_scope("val_inputs"):
                    val_input_images = transform(val_inputs, args, training=False)

                with tf.name_scope("val_gts"):
                    val_target_images = transform(val_targets, args, training=False)

                with tf.name_scope("val_batch"):
                    self._val_batch['paths'], self._val_batch['inputs'], self._val_batch['targets'] = tf.train.batch(
                        [val_paths, val_input_images, val_target_images],
                        batch_size=args.batch_size)

                self._images['inputs'] = deprocess(self._val_batch['inputs'])
                self._images['targets'] = deprocess(self._val_batch['targets'])

        """
            A placeholder to change to val batch during training
        """

        self._training = tf.placeholder(tf.bool)

        if args.train and self._validation:

            def f1(): return self._train_batch

            def f2(): return self._val_batch

            self._batch = tf.cond(self._training, f1, f2)

        elif args.train:
            self._batch = self._train_batch

        else:
            self._batch = self._val_batch

    def _create_model(self):

        args = self._args
        batch = self._batch
        eps = 1e-12

        paths, inputs, targets = batch['paths'], batch['inputs'], batch['targets']

        """
            ======================================
            Todo:
                Add your model structure code here
            ======================================
        """

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = generator(inputs, out_channels, args.ngf)



        """
            ======================================
            Todo:
                Add your losses here
            ======================================
        """

        with tf.variable_scope('losses'):
            l1_loss = L1(targets, outputs)
            l2_loss = L2(targets, outputs)

        loss = l1_loss * 100 + l2_loss



        """
            ======================================
            Todo:
                Add your optimizer & trainer here
            ======================================
        """

        with tf.variable_scope('trainer'):
            trainer = tf.train.AdamOptimizer(args.lr, args.beta1).minimize(loss)





        """
           ===================================
             save to self._ member variables
           ===================================
        """
        ema = tf.train.ExponentialMovingAverage(decay=0.99)  # MovingAverage
        update_losses = ema.apply([l1_loss, l2_loss])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        self._losses['L1_loss'] = ema.average(l1_loss)
        self._losses['L2_loss'] = ema.average(l2_loss)

        self._outputs['outputs'] = outputs
        self._train = tf.group(update_losses, incr_global_step, trainer)

    def test(self):
        args = self._args

        tfconfig = allow_gpu_growth_config()
        sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=tfconfig) as sess:
            self.restore(sess)
            start_step = sess.run(sv.global_step)
            color_print("Test on checkpoint with step: %d" % start_step, 3)

            results = self._eval(sess)
            im_paths, imgs = [], []
            for paths in results['paths']:
                for i in paths:
                    path = str(i, encoding="utf-8")
                    im_paths.append(path.split('/')[-1])

            for _images in results['images']:
                for img in _images['converted_outputs']:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)

            for path, img in list(zip(im_paths, imgs)):
                if args.output_dir:
                    save_path = os.path.join(args.output_dir, path)
                    print("Save output image '%s'" % save_path)
                    cv2.imwrite(save_path, img)
                else:
                    cv2.imshow('result', img)
                    cv2.waitKey(0)

    def train(self):
        args = self._args

        logdir = args.log_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None

        tfconfig = allow_gpu_growth_config()
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=tfconfig) as sess:
            # print parameter info
            print("Parameter Count =", sess.run(self._debug['parameter_count']))

            start_step = 0
            """
                check if loading a checkpoint is required 
            
            """
            if args.checkpoint is not None or args.resume:
                self.restore(sess)
                start_step = sess.run(sv.global_step)
                color_print("Resume training from step: %d" % start_step, 3)
                sys.stdout.flush()

            """
                calc max steps
            """
            if args.epochs is not None:
                self._max_steps = self._steps_per_epoch * args.epochs
            if args.max_steps is not None:
                self._max_steps = args.max_steps

            start = time.time()

            for step in range(start_step, self._max_steps):
                self._step = step

                """
                    construct `run_dict`
                """

                run_dict = {
                    "train": self._train,
                    "global_step": sv.global_step,
                }
                if self._is_time_to(args.progress_freq):
                    run_dict.update(self._losses)
                    run_dict['debug'] = self._debug

                if self._is_time_to(args.summary_freq):
                    run_dict['summary'] = sv.summary_op

                """
                    sess.run
                """

                results = sess.run(run_dict, feed_dict={self._training: True})

                """
                    Add Summary
                """
                if self._is_time_to(args.summary_freq):
                    sv.summary_writer.add_summary(results['summary'], results['global_step'])

                """
                    Show training progress bar
                """
                if self._is_time_to(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / self._steps_per_epoch)
                    train_step = (results["global_step"] - 1) % self._steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (self._max_steps - step) * args.batch_size / rate

                    # progress bar
                    msg = '(loss) L1:%.3f L2:%.3f | ETA: %s' % (
                        results["L1_loss"], results["L2_loss"],
                        format_time(remaining))
                    pre_msg = 'Epoch:%d ' % train_epoch
                    if train_epoch > 0 and train_step > 0:
                        progress_bar(train_step - 1, self._steps_per_epoch, pre_msg, msg)

                """
                    Save the model
                """
                if self._is_time_to(args.save_freq):
                    self.save(sess)

                """
                    Todo 
                    Add your evaluation code here
                """
                if step != 0 and step % args.eva_freq == 0:
                    self._eval(sess)
                    # self._average['ssim'] = []
                    # self._average['psnr'] = []
                    # print()
                    # sys.stdout.flush()
                    # for i in range(500//args.batch_size):
                    #     progress_bar(i, 500//args.batch_size, 'Eva.... ')
                    #     run_dict = self._eva
                    #     run_dict['debug'] = self._debug
                    #     run_dict['images'] = self._images
                    #     # sess.run to fetch results
                    #     results = sess.run(run_dict, feed_dict={self._training: False})
                    #     self._eval(results)
                    #
                    # print('psnr: %f' % np.average(self._average['psnr']))
                    # print('ssim: %f' % np.average(self._average['ssim']))

    def _eval(self, sess):
        args = self._args

        # ssim = tf.image.ssim(results['outputs'], results['targets'], max_val=1.0)
        #print()
        # outputs = results['images']['outputs']
        # targets = results['images']['targets']
        # inputs = results['images']['inputs']

        #print(results['debug']['paths'])

        # outputs = cv2.cvtColor(outputs[0], cv2.COLOR_BGR2RGB)
        # #targets = cv2.cvtColor(targets[0], cv2.COLOR_BGR2RGB)
        # inputs = cv2.cvtColor(inputs[0], cv2.COLOR_BGR2RGB)

        # cv2.imshow('inputs', inputs[0])
        # cv2.imshow('targets', targets[0])
        # cv2.imshow('outputs', outputs[0])
        # cv2.waitKey(0)
        self._average['ssim'] = []
        self._average['psnr'] = []
        evaluation = {}
        evaluation['images'] = []
        evaluation['paths'] = []
        print()
        sys.stdout.flush()
        for i in range(500 // args.batch_size):
            progress_bar(i, 500 // args.batch_size, 'Eva.... ')
            run_dict = self._eva
            run_dict['debug'] = self._debug
            run_dict['images'] = self._images
            run_dict['paths']=self._batch['paths']
            # sess.run to fetch results
            results = sess.run(run_dict, feed_dict={self._training: False})
            evaluation['images'].append(results['images'])
            evaluation['paths'].append(results['paths'])
            self._average['psnr'].append(np.average(results['psnr']))
            self._average['ssim'].append(np.average(results['ssim']))

        print('psnr: %f' % np.average(self._average['psnr']))
        print('ssim: %f' % np.average(self._average['ssim']))

        return evaluation
