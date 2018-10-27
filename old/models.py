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

from scheduler import*


# not good:
cuda='True'


def CFIAR_data_loaders(base_data_dir, train_samples_index, valid_samples_index, kwargs, batch_size=64):
    """Make train, validation, test data loaders for MNIST dataset. Limited augmnetation, nothing fancy
    
    Arguments:
        train_samples_index: index of the train samples  ???(what is this index based off???)
        valid_samples_index: index of the valid samples
        batch_size: 
    """

    class ChunkSampler(sampler.Sampler):
        """Samples elements sequentially from some offset. 
        
        Argument:
            samples_index: index of desired samples
        """
        def __init__(self, samples_index):
            self.samples_index = samples_index

        def __iter__(self):
            return iter(self.samples_index)
    
        def __len__(self):
            return len(self.samples_index)


    train_set = datasets.CIFAR10(base_data_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.RandomRotation(15),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, **kwargs,
        sampler=ChunkSampler(train_samples_index))


    valid_set = datasets.CIFAR10(base_data_dir, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size, **kwargs,
        sampler=ChunkSampler(valid_samples_index))

    test_set = datasets.CIFAR10(base_data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=True, **kwargs)

    return(train_loader, valid_loader, test_loader)



def MNIST_data_loaders(base_data_dir, train_samples_index, valid_samples_index, kwargs, batch_size=64):
    """Make train, validation, test data loaders for MNIST dataset. Limited augmnetation, nothing fancy
    
    Arguments:
        train_samples_index: index of the train samples  ???(what is this index based off???)
        valid_samples_index: index of the valid samples
        batch_size: 
    """
    class ChunkSampler(sampler.Sampler):
        """Samples elements sequentially from some offset. 
        
        Argument:
            samples_index: index of desired samples
        """
        def __init__(self, samples_index):
            self.samples_index = samples_index

        def __iter__(self):
            return iter(self.samples_index)
    
        def __len__(self):
            return len(self.samples_index)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(base_data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomRotation(15),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, **kwargs,
        sampler=ChunkSampler(train_samples_index))

    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(base_data_dir, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, **kwargs,
        sampler=ChunkSampler(valid_samples_index))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(base_data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return(train_loader, valid_loader, test_loader)


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.reset_all_parameters()

    def reset_all_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()
                # size = m.weight.size()
                # fan_out = size[0] # number of rows
                # fan_in = size[1] # number of columns
                # variance = np.sqrt(2.0/(fan_in + fan_out))
                # m.weight.data.normal_(0.0, variance)
        # stdv = 1. / math.sqrt(self.input_caps)
        # self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    
    def get_lr_performance(self, optimizer, scheduler, train_loader, valid_loader, epochs, verbose=False):        
        """Return the validation after training for given epochs"""
        
        def get_valid_loss():
            # Now get the validation loss
            self.eval()
            valid_loss = 0
            correct = 0
            num_data=0
            for data, target in valid_loader:
                num_data+=len(target)
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                output = self(data)
                valid_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            valid_loss /= num_data
            valid_acc = 100. * correct / num_data
            return valid_loss, valid_acc    
        
        for epoch in range(1, epochs + 1):
            # Every epoch step the scheduler
            scheduler.step()
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            # to deubg
            if verbose:
                valid_loss, valid_acc = get_valid_loss()
                print('-------Epoch: ', epoch)
                print('Learning Rate: ', scheduler.get_lr())
                print('train loss: ', loss.data.cpu().numpy()[0])
                print('valid_loss: ', valid_loss, 'valid_acc: ', valid_acc)
       
        # final validation loss
        valid_loss, valid_acc = get_valid_loss()
        return valid_loss, valid_acc
    
    def test(self):
        self.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_acc = 100. * correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return test_loss, test_acc

    def get_population_perf(self, population, train_loader, valid_loader, test_loader, momentum):
        """Evaluate all the schedules for epochs. Use data loaders for each. Reinitialize network for eacch schedule
        
        Args:
            population (list of lists): list schedules which are lists of length epochs
            train_loader (torch DataLoader)
            valid_loader (torch DataLoader) 
            test_loader (torch DataLoader)
            momentum (float): momentum for SGD
            
        Returns:
            pop_perf: list of tuples indicating the schedule and its' accuracy
        """
        
        epochs = len(population[0])
        
        pop_perf = []
        for curr_schedule in population:
            # make sure the weights are reinitialized before training the network
            self.reset_all_parameters() ##?????
            lr = curr_schedule[0] # is this even used?
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
            scheduler = FixedLR(optimizer=optimizer, schedule=curr_schedule)
            loss, acc = self.get_lr_performance(optimizer, scheduler, train_loader, valid_loader, epochs, verbose=False)
            pop_perf.append((acc, curr_schedule))
            
        return pop_perf
