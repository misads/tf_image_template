# encoding=utf-8
"""
BaseModel for tf_image_template

Author: xuhaoyu@tju.edu.cn

update 11.25

"""

import os
import sys

import tensorflow as tf
import utils.misc_utils as utils


class BaseModel(object):
    """
        Example:
        class MyNet(BaseModel):
            def __init__(self, args, *pipelines):
            BaseModel.__init__(self, args)

            if args.train:
                self._pipelines['train'] = pipelines[0]
                if args.val_dir:
                    self._pipelines['val'] = pipelines[1]

            elif args.test:
                self._pipelines['val'] = pipelines[0]

            self._build_graph()
    """
    def name(self):
        return self.__class__.__name__.lower()

    def __init__(self, args):
        self._args = args
        self._validation = args.test or args.val_dir is not None

        self._pipelines = {}  # network input pipelines(train/val/test)
        self._train_batch = {}
        self._val_batch = {}
        self._batch = None  # could be train or val batch

        self._train = {}  # use for sess.run(train)
        self._losses = {}  # use to trace losses
        self._eva = {}  # tensors for evaluation

        self._average = {}  # temp array to calc average score

        self._module = {}  # output of a module (e.g. a Generator)
        self._tensors = {}  # intermediate tensors
        self._images = {}  # image outputs (should be all the same type of either tf.uint8 or tf.float32)

        self._outputs = {}  # network outputs
        self._debug = {}  # debug information

        self._step = 0  # global step
        self._max_steps = 1000000

    def _create_debug_tensors(self):
        self._debug['parameter_count'] = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        # Usage:
        # print("Parameter Count =", sess.run(self._debug['parameter_count']))

        variables = tf.trainable_variables()
        variable_shapes = [tf.shape(v) for v in tf.trainable_variables()]  # or tf.global_variables

        scope = None
        self._debug['variables'] = utils.get_all_tf_variables(trainable_only=True, scope=scope)
        # Usage:
        # for k, v in sess.run(self._debug['variables']):
        #     print("   '%s' shape=%s" % (k, str(v)))

    def _manage_outputs(self):
        raise NotImplementedError

    def _manage_inputs(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _build_graph(self):
        self._manage_inputs()

        self._create_model()

        self._manage_outputs()  # summary and evaluation

        self._create_debug_tensors()  # parameters, variables, layers and so on

        self._saver = tf.train.Saver(max_to_keep=5)  # a Saver/Restorer to save/restore checkpoint

    def _eval(self, results):
        """
            use np.ndarray results to evaluate
            example:
                    eval_size = self._pipelines['eval']._size()
                    for i in range(eval_size // args.batch_size):
                        progress_bar(i, eval_size, 'Eva.... ')
                        results = sess.run(run_dict, feed_dict={self._training: False})
                        self.eval(results)
            :return:
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # used in test time, no backprop
    def test(self):
        raise NotImplementedError

    def _optimizers(self, optimizer):
        lr = self._args.lr
        if optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr)
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(lr)
        elif optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr)
        else:
            print('Unknown optimizer: %s' % optimizer)
            raise ValueError
        self._optimizer = optimizer

    def _is_time_to(self, freq):
        return freq > 0 and ((self._step + 1) % freq == 0 or self._step == self._max_steps - 1)

    def save(self, sess, label='model'):

        save_path = os.path.join(self._args.output_dir, label)

        prefix = self._saver.save(sess, save_path, global_step=self._step, write_meta_graph=False)
        if prefix:
            print("[saving checkpoint '%s']   " % prefix)
        else:
            utils.color_print("saving checkpoint failed.", 1)
        sys.stdout.flush()

    def restore(self, sess, latest=False):

        ckpt_path = None
        if self._args.checkpoint is not None:
            ckpt_path = self._args.checkpoint
            if os.path.isdir(ckpt_path):
                latest = True
            if latest:
                self._saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            else:
                s = self._saver.restore(sess, ckpt_path)  # .data文件
                print(s)

        elif self._args.resume and os.path.isdir('checkpoint'):
            ckpt_path = 'checkpoint'
            s = self._saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            print(s)
        if ckpt_path:
            utils.color_print("Checkpoint loaded from '%s'" % ckpt_path, 3)
            sys.stdout.flush()

    # def save(self, label=None):
    #     epoch = self.epoch
    #     iterations = self.iterations
    #
    #     if label is None:
    #         model_name = os.path.join(self.save_dir, self.name() + '_%03d_%08d.pt' % ((epoch), (iterations)))
    #     else:
    #         model_name = os.path.join(self.save_dir, self.name() + '_' + label + '.pt')


