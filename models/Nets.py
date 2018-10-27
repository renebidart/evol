import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetMNIST(nn.Module):
    """ No padding, on the input so 4x4. size = 28*28

    Assuming is easier to generate bounded values, so will use tanh non-linearity instead of relu
    tanh seems a bit better than sigmoid, but both degrade performance a bit. tanh probably goes better with relu around 0
    """
    def __init__(self, num_filters=20):
        super(SimpleNetMNIST, self).__init__()
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=2,  padding=2)
        self.conv2 = nn.Conv2d(15, num_filters, kernel_size=5, stride=2,  padding=2)
        self.fc1 = nn.Linear(num_filters*7*7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.view(-1, self.num_filters*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def encode_feat(self, x):
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x
