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
# import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
if __name__ == '__main__':
    alpha = 1
    dataset = 'cifar10'

    # server_path = r'\\192.168.210.100\mnt\fs1\openpai\zhangxingyan'
    Fed_AVGPath = r"\\192.168.210.100\mnt\fs1\openpai\lzhangxingyan\Fed_AVG\save"
    FedEnsPath = r"\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save"
    # print(os.path.exists(server_path+"\Fed_ensemblev2.2\save"))
 # “\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save\Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name(cifar10_alpha_0.1_P_2_M1).json”
    file_name = {

        "FecAVG_0.1":      Fed_AVGPath+ f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.1)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',
        "FecAVG_0.05": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.05)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',
        "FecAVG_0.01": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',
        "FecAVG_0.005": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.005)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',
        "FecAVG_0.001": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.001)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',

    }
    # bash run.sh cifar10 1 0.1 0 0.1
    # bash run.sh cifar10 1 0.1 0 0.05
    # bash run.sh cifar10 1 0.1 0 0.01
    # bash run.sh cifar10 1 0.1 0 0.005
    # bash run.sh cifar10 1 0.1 0 0.001
    # Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_M3_MP3_name(cifar10_alpha_100_P_2).json


    for name,file in file_name.items():
        try:
            with open(file) as f:
                f  = json.load(f)
                key = 'global_test_accs'
                key = 'test_accs'
                print(key)
                num = len(f[key])
                # for i in range(num):
                plt.plot(f[key][:], label=name)

                print(f" size = {len(f[key])}", end=' ')
                # print(f"client = {i} best_acc")
                epochs = len(f[key])
                print('last_acc = {:.4f} '.format(f[key][epochs - 1]), end=' ')
                print('best_acc = {:.4f} '.format(max(f[key])))
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
    plt.show()
    plt.savefig('test.png')
