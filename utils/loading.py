"""
???
Delete all the loading for models that are never used (all except preact)
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from models.preact_resnet import PreActResNet18
from models.Nets import SimpleNetMNIST


def load_net(model_loc, args=None):
    model_file = Path(model_loc).name
    model_name = model_file.split('-')[0]

    if (model_name == 'SimpleNetMNIST'):
        model = SimpleNetMNIST(num_filters=int(model_file.split('-')[1].split('_')[0]))
    else:
        print(f'Error : {model_file} not found')
        sys.exit(0)
    model.load_state_dict(torch.load(model_loc)['state_dict'])
    return model


# Return network & a unique file name
def net_from_args(args, num_classes, IM_SIZE):
    if (args.net_type == 'PreActResNet18'):
        net = PreActResNet18()
        file_name = 'PreActResNet18-'
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes, IM_SIZE)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, IM_SIZE)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'SimpleNetMNIST'):
        net = SimpleNetMNIST()
        file_name = 'SimpleNetMNIST'
    else:
        # print('Error : Wrong net type')
        sys.exit(0)
    return net, file_name


# def load_net_cifar(model_loc):
#     """ Make a model
#     Network must be saved in the form model_name-depth, where this is a unique identifier
#     """
#     model_file = Path(model_loc).name
#     model_name = model_file.split('-')[0]
#     print('Loading model_file', model_file)
#     if (model_name == 'vggnet'):
#         model = VGG(int(model_file.split('-')[1]), 10)
#     elif (model_name == 'resnet'):
#         model = ResNet(int(model_file.split('-')[1]), 10)
#     # so ugly
#     elif (model_name == 'preact_resnet'):
#         if model_file.split('/')[-1].split('_')[2] == 'model': 
#             model = PreActResNet(int(model_file.split('-')[1].split('_')[0]), 10)
#         else:
#             model = PResNetReg(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)

#     elif (model_name == 'wide'):
#         model = Wide_ResNet(model_file.split('-')[2][0:2], model_file.split('-')[2][2:4], 0, 10, 32)
    
#     # Dumb ones
#     elif (model_name == 'PResNetRegNoRelU'):
#         model = PResNetRegNoRelU(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)
    
#     else:
#         print(f'Error : {model_file} not found')
#         sys.exit(0)
#     model.load_state_dict(torch.load(model_loc)['state_dict'])
#     return model