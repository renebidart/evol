import torch
from torch.optim import Optimizer

# ???
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data
# import torch.utils.data.sampler as sampler

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
class ListScheduler(_LRScheduler):
    """Learning rate for each epoch corresponds to a learning rate in the list. No decay.
    """

    def __init__(self, optimizer, schedule, last_epoch=-1):
        if not type(schedule) == list:
            raise ValueError('Schedule should be a list of'
                             ' floats. Got {}', schedule)
        self.schedule = schedule
        super(ListScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule[self.last_epoch]]

    def get_rate(self, epoch=None, num_epoches=None):
        self.trn_iterations += 1
        self.clr_iterations += 1
        lr = self.clr()
        return lr