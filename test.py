import torch.nn as nn
import torch
# m = torch.ones([3,1])
# print(a)
a = torch.ones(2,1,10)
b = torch.tensor([1,2])
b = b.resize(2,1,1)
c = torch.multiply(a,b)
print(c)