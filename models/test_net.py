"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['Net']

class Net(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.quant1 = quant()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.quant2 = quant()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.quant3 = quant()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.quant4 = quant()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.quant1(self.relu1(self.conv1(x))))
        x = self.pool(self.quant2(self.relu2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.quant3(self.relu3(self.fc1(x)))
        x = self.quant4(self.relu4(self.fc2(x)))
        x = self.fc3(x)
        return x

class MBase:
    base = Net
    args = list()
    kwargs = dict()

class Net(MBase):
    pass

# net = Net()