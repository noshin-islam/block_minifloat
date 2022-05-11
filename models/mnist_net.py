"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['MNet', 'MNetSmall']

class MNet(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5) #nn.Conv2d(3, 6, 5) edit -> (20 - 5)/1 + 1 = 16
        self.relu1 = nn.ReLU()
        # maxpool 2,2 = 8  -> (16-2)/2+1 = 8
        self.conv2 = nn.Conv2d(3, 6, 5)    #(8-5)/1+1 = 4
        self.relu2 = nn.ReLU()
        # maxpool2,2 = 2 -> (4-2)/2+1 = 2
        self.fc1 = nn.Linear(6 * 2 * 2, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.quant = quant()

    def forward(self, x):
        
        x = self.quant(x)
        # print("POST QUANT DATA  ", x)   ### CULPRIT FOUND
        out = self.pool(self.relu1(self.conv1(x)))
        # print("SHAPE1: ", out.shape)

        # out = self.pool(out)
        # print("POOL DATA NAN? ", out.isnan().any())

        # print("OUT0 POST LAYERS: ", out)
        
        # print("OUT1 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.pool(self.relu2(self.conv2(out)))
        # print("OUT1 POST QUANT: ", out)
        # print("HERE:", out.size(0))
        # print("SHAPE2: ", out.shape)
        out = out.view(out.size(0), -1)
        # print("SHAPE3: ", out.shape)
        # print("OUT2 PRE QUANT: ", out)
        out = self.quant(out)
        out = self.relu3(self.fc1(out))
        # print("SHAPE4: ", out.shape)
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

class MNetSmall(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, 1) #nn.Conv2d(3, 6, 5) edit -> (20 - 5)/1 + 1 = 16
        # 4x4 -> (4-1)/1 + 1 = 4
        # (15-3)/1+1 = 13
        self.relu1 = nn.ReLU()
        
        self.fc1 = nn.Linear(2 * 4 * 4, num_classes)
    
        self.quant = quant()

    def forward(self, x):
        print("INPUT SHAPE: ", x.shape)
        out = self.quant(x)
        out = self.conv1(out)
        out = self.relu1(out)
        print("HERE: ",out.shape)
        out = out.view(out.size(0), -1)
        print("HERE1: ", out.shape)
        out = self.quant(out)
        out = self.fc1(out)
    
        return out

class MnistBase:
    base = MNet
    args = list()
    kwargs = dict()

class MNet(MnistBase):
    pass

class MNetSmall(MnistBase):
    base = MNetSmall

# net = Net()