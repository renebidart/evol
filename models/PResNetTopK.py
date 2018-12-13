import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SpatialTopK


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = activation

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out) if hasattr(self, 'activation') else out
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.bn2(x)
        out = self.activation(out) if hasattr(self, 'activation') else out
        out = self.conv2(out)
        out += shortcut
        return out
    

# Not updated yet
# class PreActBottleneck(nn.Module):
#     '''Pre-activation version of the original Bottleneck module.'''
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out += shortcut
#         return out


class PResNetTopK(nn.Module):
    def __init__(self, block, num_blocks, within_block_act=None, after_block_act=None, 
                 frac_list=[.25, .25, .25, .25], group_list=[1, 1, 1, 1], num_classes=10):
        super(PResNetTopK, self).__init__()
        """
        Same top k fractions and groups used within the entire "layer" and the after_bloc_act
        Only 4 fracs/groups need to be defined no matter searching over after or within or both
        
        Based off PreActResNet Arch, reference:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
            Identity Mappings in Deep Residual Networks. arXiv:1603.05027
        Implementation Based off: 
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
        
        To make PResNet > 34 must add the bottleneck layer, not implemented here.
        Don't use within and between activations that are the same. It is duplicated computation
        because there is an activation at the start of every "block/layer"
        """

        self.after_block_activations = []
        for i, block_size in enumerate(num_blocks):
            self.after_block_activations.append(get_activation(after_block_act, frac=frac_list[i], groups=group_list[i]))

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, get_activation(within_block_act, frac=frac_list[0], groups=group_list[0]))
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, get_activation(within_block_act, frac=frac_list[1], groups=group_list[1]))
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, get_activation(within_block_act, frac=frac_list[2], groups=group_list[2]))
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, get_activation(within_block_act, frac=frac_list[3], groups=group_list[3]))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x): # ugly maybe switch to a loop
        out = self.conv1(x)
        out = self.activations[0](out) if self.after_block_activations[0] else out
        out = self.layer1(out)
        out = self.activations[1](out) if self.after_block_activations[1] else out
        out = self.layer2(out)
        out = self.activations[2](out) if self.after_block_activations[2] else out
        out = self.layer3(out)
        out = self.activations[3](out) if self.after_block_activations[3] else out
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    
def get_activation(activation, frac=.25, groups=1):
    if (activation == 'relu'):
        activation = nn.ReLU()
    elif (activation == 'topk'):
        activation = SpatialTopK(topk=1, frac=frac, groups=groups)
    else:
        print('No Activation')
        activation = None
    return activation
    


# Presnet 18 has 4 blocks with layers: [2,2,2,2] and corresponding num features: [64, 128, 256, 512]
class PResNetTopK18(PResNetTopK):
    def __init__(self, block, num_blocks, within_block_act, after_block_act, 
                 frac_list, group_list, num_classes=10):
        super(PResNetTopK18, self).__init__()
        self.block = PreActBlock
        self.num_blocks = [2,2,2,2]
        self.within_block_act = within_block_act
        self.after_block_act = after_block_act
        self.frac_list = frac_list
        self.group_list = group_list
        self.num_classes = num_classes




# def PResNetTopK18():
#     return TestNet(block=PreActBlock, block_sizes=[2,2,2, 2], block_features=[64, 128, 256], use_residual=False, within_block_activation='relu', 
#                    after_block_activation=None, frac=.25, groups=1, num_classes=10)

    
    
# def TestNetNotPResNet18():
#     return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=False, within_block_activation='relu', 
#                    after_block_activation=None, frac=.25, groups=1, num_classes=10)

# def TestNetMostlyPResNet18():
#     return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, within_block_activation='relu', 
#                    after_block_activation=None, frac=.25, groups=1, num_classes=10)

# def TestNetPResnet18TopK(): # .05 is about 91% with normal PResNet
#     return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, within_block_activation='relu', 
#                    after_block_activation='topk', frac=.05, groups=1, num_classes=10)

# def TestNetPResnet18TopKEverywhere(): # .05 is about 91% with normal PResNet
#     return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, within_block_activation='topk', 
#                    after_block_activation='topk', frac=.05, groups=1, num_classes=10)

# def TestNetPResnet18TopK_act(): # .05 is about 91% with normal PResNet
#     return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, within_block_activation=None, 
#                    after_block_activation='topk', frac=.05, groups=1, num_classes=10)
 
    
    
    


# class Block(nn.Module):
#     '''Pre-activation version of the BasicBlock.

#     First conv uses the stride, rest are stride 1
#     same number of features for each layer
#     Shortcut: Uses 1x1 conv with stride to adjust for more layers or a stride

#     bn->act->conv->bn->act->conv ...
#     Shortcut is adjusting the residual for downsampling spatially(stride) or increasing channels
#     Always use_residual
#     '''

#     def __init__(self, in_planes, planes, stride, num_layers, use_residual=True, activation=None, frac=.25, groups=1):
#         super(Block, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.use_residual = use_residual
#         self.conv_all = []
        
#         for i in range(1, num_layers):
#             self.conv_all.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
#             )

#         if activation:
#             self.activation = get_activation(activation)

#     def forward(self, x):
#         out_pre_conv = self.bn1(x)
#         out_pre_conv = self.activation(out_pre_conv) if hasattr(self, 'activation') else out_1
#         out = self.conv1(out_pre_conv)

#         for key, conv_layer in enumerate(self.conv_all):
#             out = self.bn2(out)
#             out = self.activation(out) if self.activation else x
#             out = conv_layer(out)

#         if self.use_residual:
#             shortcut = self.shortcut(out_pre_conv) if hasattr(self, 'shortcut') else x
#             out += shortcut
#         return out



# class TestNet(nn.Module):
#     def __init__(self, block_sizes, block_features, use_residual=True, within_block_activation=None, 
#                  after_block_activation=None, frac_list=[.25, .25, .25], groups=1, num_classes=10):
#         super(TestNet, self).__init__()
#         """
#         For testing activations within and at end of blocks.
        
#         Based off PreActResNet Arch, reference:
#             Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#             Identity Mappings in Deep Residual Networks. arXiv:1603.05027
#         Implementation Based off: 
#         https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
        
#         To make PResNet > 34 must add the bottleneck layer, not implemented here.
#         This messy loop to add layers/blocks is to use variable number of blocks
#         """
        
#         self.layers = []
#         self.activations = []

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
#         in_planes = 64
#         for i, block_size in enumerate(block_sizes):
#             self.layers.append(Block(in_planes, block_features[i], num_layers=block_size, 
#                                      stride=2, use_residual=use_residual, activation=within_block_activation, 
#                                      frac=frac, groups=groups))
#             self.activations.append(get_activation(after_block_activation, frac=.25, groups=1))
#             in_planes = block_features[i]

#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             x = self.activations[i](x) if self.activations[i] else x

#         out = F.adaptive_avg_pool2d(x, (1, 1))
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out