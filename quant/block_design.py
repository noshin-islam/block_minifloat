import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# block design implementations
# block_type = "w" weight, "x" activation

# calc_padding
def calc_padding(fold, dim):
    if fold>=dim:
        nstack = 1
    else:
        quo,rem = divmod(dim, fold)
        nstack = quo+(rem>0)
    num = nstack*fold
    p = num-dim
    return int(p)

# block entire vector
def block_V(data, ebit, func, k_exp):
    
    entry = func(torch.abs(data), 0) #.item() ----- extracts maximum from each column, but there are no other cols, so its just max w 0
    
    if entry == 0: return data

    shift_exponent = torch.floor(torch.log2(entry+1e-28))

    # shift_exponent = torch.floor(torch.log2(entry+1e-10))
    # print("shift exp: ", shift_exponent)
    # print("shift exp old: ", shift_exponent_old)

    min_exp = (-2**(ebit-1)) 
    max_exp = (2**(ebit-1)-1) 

    shift_exponent = torch.clamp(shift_exponent, min_exp, max_exp)

    ####THIS IS THE OLD ONE
    # shift_exponent = torch.clamp(shift_exponent, -2**(ebit-1), 2**(ebit-1)-1)

    return shift_exponent
    #entry = func(torch.abs(data), 0).item()
    #if entry == 0: return data
    #shift_exponent = math.floor(math.log2(entry))
    #shift_exponent = min(max(shift_exponent, -2**(ebit-1)), 2**(ebit-1)-1)
    #return shift_exponent


# block on axis=0  -- block based on row
# shift exp returned will have same rows as mantissa but 1 as the other dims as each row of mantissa gets one exp

def block_B(data, ebit, func, k_exp):

    print("here from block b")
    # print("data:", data)
    entry = func(torch.abs(data.view(data.size(0), -1)), 1)#[0] ####extracts the maximum value present in each column of data.
    # print("entry: ", entry)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zero_mat = torch.zeros(data.shape).to(device)
    _, count = (torch.unique(data[zero_mat == data], return_counts=True))
    # print(torch.numel(a))
    # print(count)
    if (count.nelement() != 0 and count >= torch.numel(data)/2):
        print("0 tensor detected!")
        shift_exponent = torch.floor(torch.log2(entry+1e-10))
        # return data
    else:
        shift_exponent = torch.floor(torch.log2(entry+1e-28))

    
    print("shift exp: ", shift_exponent)
    # print("shift exp old: ", shift_exponent_old)
    
    #clamping the shift exponent between the highest representable values eg, -128 to 127 for 8 bit ebit.
    # shift_exponent = torch.clamp(shift_exponent, -2**(ebit-1), 2**(ebit-1)-1)

    min_exp = (-2**(ebit-1)) 
    max_exp = (2**(ebit-1)-1) 
    print("min: ", min_exp, "max: ", max_exp)

    shift_exponent = torch.clamp(shift_exponent, min_exp, max_exp)
    print("shift exp post clamp ", shift_exponent)

    shift_exponent = shift_exponent.view([data.size(0)]+[1 for _ in range(data.dim()-1)]) #[no. of rows, 1 , 1, 1, 1, 1, 1 .... 1]
    # print("shift exp output ", shift_exponent)
    #example shape of shift_exponent -> [64, 1, 1, 1]
    # print("Block B shift exp ", shift_exponent)
    # print("size ", shift_exponent.shape)

    return shift_exponent

# block entire tensor
def block_B0(data, ebit, func, k_exp):
    entry = func(torch.abs(data.view(-1)), 0).item()
    if entry == 0: return data
    shift_exponent = math.floor(math.log2(entry))

    ## THIS WORKS
    # shift_exponent = min(max(shift_exponent, -2**(ebit-1)), 2**(ebit-1)-1)

    min_exp = (-2**(ebit-1)) 
    max_exp = (2**(ebit-1)-1) 
    shift_exponent = min(max(shift_exponent, min_exp), max_exp)
    

    return shift_exponent


#"""
# block by some factor on axis=0,1 (dim=2)
def block_BG2(data, factors, ebit, func, k_exp):
    dim = data.size()
    assert len(dim) == 2

    # factors already 2D
    fact = [factors[i] if factors[i] != -1 else dim[i] for i in range(2)]

    # pad each dimension
    num_pad = [calc_padding(fact[i], dim[i]) for i in range(2)]
    padding =tuple([0,num_pad[1],0,num_pad[0]])
    data = F.pad(input=data, pad=padding, mode='constant', value=0)
    dim_pad = data.size()

    # unfold ---splitting the data into blocks/tiles???
    data_unf = data.unfold(0, fact[0], fact[0]).unfold(1, fact[1], fact[1])

    # calc shift_exponent for block
    data_f = data_unf.contiguous().view([data_unf.size(0), data_unf.size(1),-1])
    tiles = data_f.size()[:2]
    mean_entry = func(torch.abs(data_f), 2) #[0]
    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zero_mat = torch.zeros(data_f.shape).to(device)
    _, count = (torch.unique(data_f[zero_mat == data_f], return_counts=True))
    # print(torch.numel(a))
    # print(count)
    if (count.nelement() != 0 and count >= torch.numel(data_f)/2):
        # print("0 tensor detected!")
        shift_exponent = torch.floor(torch.log2(mean_entry+1e-10))
        # return data
    else:
        shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))




    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))

    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-10))
    # print("shift exp: ", shift_exponent)
    # print("shift exp old: ", shift_exponent_old)
    ### THIS WORKS
    # shift_exponent = torch.clamp(shift_exponent, -2**(ebit-1), 2**(ebit-1)-1)

    min_exp = (-2**(ebit-1)) 
    max_exp = (2**(ebit-1)-1) 
    shift_exponent = torch.clamp(shift_exponent, min_exp, max_exp)

    # reverse the unfold
    shift_exponent = shift_exponent.repeat(fact[0],fact[1])
    shift_exponent = shift_exponent.view(fact[0],tiles[0],fact[1],tiles[1])
    shift_exponent = shift_exponent.transpose(0,1).transpose(2,3)
    shift_exponent = shift_exponent.contiguous().view([dim_pad[0],dim_pad[1]])

    # remove the padding
    shift_exponent = shift_exponent[:dim[0],:dim[1]]

    return shift_exponent


# block by some factor on axis=0,1,2 (data_dim=4)
def block_BG4(data, factors, ebit, func, k_exp):

    # import pdb; pdb.set_trace()

    _dim = data.size()
    assert len(_dim) == 4
    
    # always fold last two dims (in BCHW order)
    data = data.view([_dim[0], _dim[1], -1])
    dim = data.size()
    fact = [factors[i] if factors[i] != -1 else dim[i] for i in range(3)]

    # pad each dimension
    num_pad = [calc_padding(fact[i], dim[i]) for i in range(3)]
    padding =tuple([0,num_pad[2],0,num_pad[1],0,num_pad[0]])
    data = F.pad(input=data, pad=padding, mode='constant', value=0)
    dim_pad = data.size()

    # unfold
    data_unf = data.unfold(0, fact[0], fact[0]).unfold(1, fact[1], fact[1]).unfold(2, fact[2], fact[2])

    # calc shift_exponent for block
    data_f = data_unf.contiguous().view([data_unf.size(0), data_unf.size(1), data_unf.size(2),-1])
    tiles = data_f.size()[:3]
    mean_entry = func(torch.abs(data_f), 3)#[0]
    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zero_mat = torch.zeros(data_f.shape).to(device)
    _, count = (torch.unique(data_f[zero_mat == data_f], return_counts=True))
    # print(torch.numel(a))
    # print(count)
    if (count.nelement() != 0 and count >= torch.numel(data_f)/2):
        # print("0 tensor detected!")
        shift_exponent = torch.floor(torch.log2(mean_entry+1e-10))
        # return data
    else:
        shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))


    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-28))



    # shift_exponent = torch.floor(torch.log2(mean_entry+1e-10))
    # print("shift exp: ", shift_exponent)
    # print("shift exp old: ", shift_exponent_old)

    # shift_exponent = torch.clamp(shift_exponent, -2**(ebit-1), 2**(ebit-1)-1)

    min_exp = (-2**(ebit-1)) 
    max_exp = (2**(ebit-1)-1) 
    shift_exponent = torch.clamp(shift_exponent, min_exp, max_exp)

    # reverse the unfold
    shift_exponent = shift_exponent.repeat(fact[0],fact[1],fact[2])
    shift_exponent = shift_exponent.view(fact[0],tiles[0],fact[1],tiles[1],fact[2],tiles[2])
    shift_exponent = shift_exponent.transpose(0,1).transpose(2,3).transpose(4,5)
    shift_exponent = shift_exponent.contiguous().view([dim_pad[0],dim_pad[1],dim_pad[2]])

    # remove the padding
    shift_exponent = shift_exponent[:dim[0],:dim[1],:dim[2]]

    # resize to dim=4 tensor
    shift_exponent = shift_exponent.view(_dim)

    return shift_exponent


def block_BFP(data, ebit, tensor_type, block_factor, func, k_exp):
    data_dim = data.dim()

    # block_factor = tile
    f0,f1 = int(block_factor),int(block_factor)

    # default (-1 means the whole dimension shares one exponent)
    p0,p1,p2 = 1,1,-1 # same as BC

    # decode
    if tensor_type == "x":
        if data_dim == 2:
            p0,p1 = f1,f0
        elif data_dim == 4:
            p0,p1,p2 = 1,f0,f1
    elif tensor_type == "w": #no backward quant
        if data_dim == 2:
            p0,p1 = f1,f0
        elif data_dim == 4:
            p0,p1,p2 = f1,f0,-1
    else:
        raise ValueError("Invalid tensor_type option {}".format(tensor_type))

    if data_dim == 2:
        if data.size()[1]<block_factor:
            shift_exponent = block_B(data, ebit, func, k_exp)
        else:
            shift_exponent = block_BG2(data, [p0,p1], ebit, func, k_exp)
    elif data_dim == 4:
        if data.size()[1]<block_factor:
            shift_exponent = block_B(data, ebit, func, k_exp)
        else:
            shift_exponent = block_BG4(data, [p0,p1,p2], ebit, func, k_exp) 
    else:
        raise ValueError("Invalid data_dim option {}".format(data_dim)) 
    
    return shift_exponent

#"""

#used to compute max exponent
#k is the scaled exponent

def block_design(data, tile, tensor_type, func, k):

    assert data.dim() <= 4 #.dim() returns the dimensionality of the data - 2D or 3D etc
    dim_threshold = 1

    if (k==0):
      ebit = 8
    else:
      ebit = 8  #exponent bits??

    # ebit = 2  #exponent bits??
    # import pdb; pdb.set_trace()
    # shift_exponent = 0
    if tile == -1:
        if data.dim() <= dim_threshold:
            shift_exponent = block_V(data, ebit, func, k)
        else:
            shift_exponent = block_B(data, ebit, func, k)

        if (k != 0):
            # print("max exp pre scaling: ", shift_exponent)
            # print("k used for scaling: ", k)
            # print("2^k used for scaling: ", 2**k)
            shift_exponent = shift_exponent * (2**k)
            # print("max exp post scaling: ", shift_exponent)

    elif tile == 0:
        shift_exponent = block_B0(data, ebit, func, k)
        if (k != 0):
            # print("max exp pre scaling: ", shift_exponent)
            # print("k used for scaling: ", k)
            # print("2^k used for scaling: ", 2**k)
            shift_exponent = shift_exponent * (2**k)
            # print("max exp post scaling: ", shift_exponent)

    else:
        if data.dim() <= dim_threshold:
            shift_exponent = block_V(data, ebit, func, k)
            # if (k != 0):
            #     shift_exponent = shift_exponent * (2**k)
        else:
            shift_exponent = block_BFP(data, ebit, tensor_type, tile, func, k)
        # print("without exponent scaling: ", shift_exponent)
        if (k != 0):
            # print("max exp pre scaling: ", shift_exponent)
            # print("k used for scaling: ", k)
            # print("2^k used for scaling: ", 2**k)
            # print("exp pre scale: ", shift_exponent)
            shift_exponent = shift_exponent * (2**k)
            # print("exp post scale: ", shift_exponent)
            # print("max exp post scaling: ", shift_exponent)
            # print("with exponent scaling: ", shift_exponent)

    return shift_exponent





 
