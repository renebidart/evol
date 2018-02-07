import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.sampler as sampler

from torch.optim import Optimizer

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
            
class FixedLR(_LRScheduler):
    """Learning rate for each epoch corresponds to a learning rate in the list. No decay.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedule (list): List of learning rates. Should be the same as nuber of epochs
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming 3 epochs, schedule = [0.05, 0.005, 0.0005]
        >>> # lr = 0.05     if epoch = 1
        >>> # lr = 0.005    if epoch = 2
        >>> # lr = 0.0005   if epoch = 3
        >>> scheduler = FixedLR(optimizer, schedule = [0.05, 0.005, 0.0005])
        >>> for epoch in range(3):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, schedule, last_epoch=-1):
        if not type(schedule) == list:
            raise ValueError('Schedule should be a list of'
                             ' floats. Got {}', schedule)
        self.schedule = schedule
        super(FixedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.schedule[self.last_epoch]