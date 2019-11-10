import os
import torch
import utils.utils as utils


class BaseModel(object):
    def name(self):
        return self.__class__.__name__.lower()

    def __init__(self, hparams):
        opt = hparams
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.traing = opt.traing
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self._count = 0

    def set_input(self, input):
        self.input = input

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

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

        torch.save(self.state_dict(), model_name)

    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            utils.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            utils.set_opt_param(optimizer, 'weight_decay', self.opt.wd)
