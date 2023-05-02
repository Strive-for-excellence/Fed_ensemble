import shelve
import torch
from matplotlib import  pyplot as plt
# from models.Nets import CNNMnist_Transfer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import json
import matplotlib as mpl
# from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
alpha = [100,1,0.1]
a = [[63,67.00,70.3,72.2,72.3],
     [59.9,63.9,68.8,69,68.9],
     [54.4,60.2,61.8,62.3,63]]
x_label = [1,2,3,4,5]
for i,item in enumerate(a):
    plt.plot(x_label,item,label = str(alpha[i]))
    plt.xticks(x_label)
plt.xlabel('模型数量')
plt.ylabel('准确率')
plt.legend()
plt.show()
