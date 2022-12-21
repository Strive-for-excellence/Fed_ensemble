import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import  rand
a = torch.arange(6).reshape(1,2,3)
b = torch.arange(6,12).reshape(1,2,3)
ls = [a,b]

key = 'res2.conv_bn_relu_1.0.convs.2.bias'
pos1 = key.find('convs')
pos2 = key.find('.',pos1+6)
print(pos1)
print(key[pos1+6:pos2])
# pos2 = key.find('.', pos1+1)
# idx = int(key[pos1 + 1:pos2])
# print(pos1)
# print(pos2)
# print(key[pos1+1:pos2])
# print(idx)