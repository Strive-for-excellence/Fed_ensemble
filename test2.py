import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import  rand
a = torch.arange(6).reshape(1,2,3)
b = torch.arange(6,12).reshape(1,2,3)
ls = [a,b]

key = 'conv2.1000.bias'
pos1 = key.find('.')
pos2 = key.find('.', pos1+1)
idx = int(key[pos1 + 1:pos2])
print(pos1)
print(pos2)
print(key[pos1+1:pos2])
print(idx)