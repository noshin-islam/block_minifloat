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
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.quant = quant()

    def forward(self, x):
        x = self.quant(x)
        out = self.pool(self.relu1(self.conv1(x)))
        
        out = self.quant(out)
        out = self.pool(self.relu2(self.conv2(out)))
        
        out = out.view(out.size(0), -1)

        out = self.quant(out)
        out = self.relu3(self.fc1(out))

        out = self.quant(out)
        out = self.relu4(self.fc2(out))

        out = self.quant(out)
        out = self.fc3(out)
        return out

class MBase:
    base = Net
    args = list()
    kwargs = dict()

class Net(MBase):
    pass

# net = Net()