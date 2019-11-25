# encoding=utf-8
import os
import tensorflow as tf
import utils.misc_utils as utils


class BaseModel(object):
    def name(self):
        return self.__class__.__name__.lower()

    def __init__(self, args):
        self._args = args

        self._pipelines = {}  # network input pipelines(train/val/test)
        self._train_batch = {}
        self._val_batch = {}

        self._train = {}  # use for sess.run(train)
        self._losses = {}  # use to trace losses
        self._eva = {}  # usr for evaluation

        self._module = {}  # output of a module (e.g. a Generator)
        self._tensors = {}  # intermediate tensors
        self._images = {}  # image outputs (should be all the same type of either tf.uint8 or tf.float32)

        self._outputs = {}  # network outputs
        self._debug = {}  # debug information

        self._step = 0  # global step
        self._max_steps = 1000000

        self._build_graph()

    def _manage_debug(self):
        pass

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

        self._manage_debug()  # parameters, variables, layers and so on

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

    # used in test time, no backprop
    def test(self):
        pass

    def _init_optimizer(self, optimizer):
        learning_rate = self.learning_rate
        if optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate)
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            print('Unknown optimizer: %s' % optimizer)
            raise ValueError
        self.__optimizer = optimizer

    def _is_time_to(self, freq):
        return freq > 0 and ((self._step + 1) % freq == 0 or self._step == self._max_steps - 1)

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def print_optimizer_param(self):
        # for optimizer in self.optimizers:
        #     print(optimizer)
        print(self.optimizers[-1])

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.save_dir, self.name() + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.save_dir, self.name() + '_' + label + '.pt')


