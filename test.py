
import argparse
import time
import torch
import torch.nn.functional as F
import utils
# import tabulate
import bisect
import os
import sys
from functools import partial
import collections
import models
from data import get_data
#from qtorch.optim import OptimLP
from optim import OptimLP
from torch.optim import SGD
from quant import *
#from qtorch import FloatingPoint
# from prettytable import PrettyTable
import sys

a = torch.tensor([[[[0.0078, 0.1765, 0.1922, 0.0078],
          [0.0314, 0.1412, 0.3294, 0.0392],
          [0.0314, 0.3529, 0.4667, 0.0510],
          [0.0196, 0.1490, 0.1608, 0.0706]]]])

print(a.shape)

b = torch.zeros(a.shape)
c = torch.zeros(a.shape)
print(c)

# if (torch.eq(a,b)):
#     print("hi")
# print(b.shape)
# print(torch.eq(a,b))
# print(torch.unique(a[a == b], return_counts=True))
_, count = (torch.unique(b[c == b], return_counts=True))
print(torch.numel(a))
if (count == torch.numel(a)):
    print("hi")
