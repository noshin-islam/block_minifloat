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
        self.conv1 = nn.Conv2d(3, 6, 5) #nn.Conv2d(3, 6, 5) edit -> (20 - 5)/1 + 1 = 16
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
        # print(self.quant)
        # print("INPUT DATA TO NET: ", x)
        # print("INPUT DATA NAN? ", x.isnan().any())
        # print("CONV1 WEIGHTS: ", self.conv1.weight)
        # print("CONV1 BIAS: ", self.conv1.bias)

        x = self.quant(x)
        # print("POST QUANT DATA  ", x)   ### CULPRIT FOUND
        out = self.pool(self.relu1(self.conv1(x)))

        # out = self.conv1(x)
        # print("CONV1 DATA NAN? ", out.isnan().any())

        # out = self.relu1(out)
        # print("RELU1 DATA NAN? ", out.isnan().any())

        # print("CONV2 WEIGHTS: ", self.conv2.weight)
        # print("CONV2 WEIGHTS GRAD: ", self.conv2.weight.grad)
        # print("CONV2 BIAS: ", self.conv2.bias)

        # out = self.pool(out)
        # print("POOL DATA NAN? ", out.isnan().any())

        # print("OUT0 POST LAYERS: ", out)
        
        # print("OUT1 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.pool(self.relu2(self.conv2(out)))
        # print("OUT1 POST QUANT: ", out)
        
        out = out.view(out.size(0), -1)

        # print("OUT2 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.relu3(self.fc1(out))
        # print("OUT2 POST QUANT: ", out)

        # print("OUT3 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.relu4(self.fc2(out))
        # print("OUT3 POST QUANT: ", out)

        # print("OUT4 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.fc3(out)
        # print("OUT FINAL LAYER POST QUANT: ", out)
        return out

class MBase:
    base = Net
    args = list()
    kwargs = dict()

class Net(MBase):
    pass

# net = Net()