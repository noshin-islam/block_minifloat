
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

# parser = argparse.ArgumentParser(description='Block Minifloat SGD training')

# parser.add_argument('--model', type=str, default="ResNet18LP", required=True, metavar='MODEL',
#                         help='model name (default: ResNet18LP)')


x = torch.from_numpy(np.arange(1, 17).reshape(4,4))
print(x)

k=0
x_bm = BlockMinifloat(exp=2, man=5, tile=0, flush_to_zero=False)
print(x_bm.emax)
print(x_bm.emin)
print(x_bm.max_number)

x_quantizer = quantizer(k, forward_number=x_bm, forward_rounding='stochastic')
print(x_quantizer)

data = x_quantizer(x)

data2 = block_minifloat_quantize(x, x_bm, rounding="stochastic", tensor_type="x", k_exp=0)
print(data)
print(data2)

 