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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

x = [0.1,0.5,1]
data = {"Individual":[0.884,0.778,0.699],"FedAVG":[0.662,0.796,0.772],"FedPer":[0.864,0.827,0.774],"FedEns":[0.878,0.817,0.766]}
for key,value in data.items():
    plt.plot(x,value,label=key)
    # plt.legend(key)
plt.legend()
plt.show()
