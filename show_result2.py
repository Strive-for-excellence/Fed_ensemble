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
if __name__ == '__main__':
    alpha = 0.1
    dataset = 'cifar100'
    # cifar100 0.05 2 3
    file_name = {

        "FedAVG":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset({dataset})_N(32)_E(1)_trainnum(500)_P(1)_lr(0.01)_name({dataset}_alpha_d_{alpha}_P_1).json',
        "FedEns":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset({dataset})_N(32)_E(1)_trainnum(500)_P(4)_lr(0.01)_name({dataset}_alpha_d_{alpha}_P_4).json',
    }


    for name,file in file_name.items():
        try:
            with open(file) as f:
                f  = json.load(f)
                key = 'global_test_accs'
                num = len(f[key])
                # for i in range(num):
                plt.plot(f[key][:], label=name)

                print(f" size = {len(f[key])}", end=' ')
                # print(f"client = {i} best_acc")
                epochs = len(f[key])
                print('last_acc = {:.3f} '.format(f[key][epochs - 1]), end=' ')
                print('best_acc = {:.3f} '.format(max(f[key])))
        except:
            print('no '+file)
            pass
            
            # print(f['test_acc'][200])
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    # plt.title('mnist dataset dirichlet alpha=100')
    plt.title(f'alpha = {alpha}')
    # plt.title('{dataset} CNN alpha = 0.5')
    # plt.show()
    plt.savefig('test.png')
