import shelve
import torch
from matplotlib import pyplot as plt
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
    alpha = 100
    dataset = 'cifar10'

    # server_path = r'\\192.168.210.100\mnt\fs1\openpai\zhangxingyan'
    Fed_AVGPath = r"\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_AVG\save"
    FedEnsPath = r"\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save"
    # print(os.path.exists(server_path+"\Fed_ensemblev2.2\save"))
    # “\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save\Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name(cifar10_alpha_0.1_P_2_M1).json”
    file_name = {
        #                                          Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(0)_lr(0.01)_model(LeNet)_name(cifar10_alpha_d_100_P_0_D_1000).json
        # "Individual": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(0)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_0_D_1000).json',
        "cifar10 独立同分布": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{100}_P_1_D_1000).json',
        # "FecAVG 数据异质性中度": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{1}_P_1_D_1000).json',
        "cifar10 非独立同分布": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{0.1}_P_1_D_1000).json',
        "cifar100 独立同分布": Fed_AVGPath + f'/Result_dataset(cifar100)_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name(cifar100_alpha_d_{100}_P_1_D_1000).json',
        # "FecAVG 数据异质性中度": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{1}_P_1_D_1000).json',
        "cifar100 非独立同分布": Fed_AVGPath + f'/Result_dataset(cifar100)_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name(cifar100_alpha_d_{0.1}_P_1_D_1000).json',
        # "FedPer": Fed_AVGPath + f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_2_D_1000).json',
        # "FedEns1": r'\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save\Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name(cifar10_alpha_0.1_P_2_M1).json',

        # "FedEns1": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name({dataset}_alpha_{alpha}_P_2_M1).json',
        # "FedEns2": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M2_MP2_name({dataset}_alpha_{alpha}_P_2_M2).json',
        # "FedEns5": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M5_MP5_name({dataset}_alpha_{alpha}_P_2_M5).json',

    }
    # Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_M3_MP3_name(cifar10_alpha_100_P_2).json
    fig,ax = plt.subplots(nrows=1,ncols=2)
    tot = 0
    for name, file in file_name.items():
        tot += 1
        with open(file) as f:
            f = json.load(f)
            key = 'global_test_accs'
            # key = 'test_accs'
            print(name)
            num = len(f[key])
            # for i in range(num):
            if tot == 1:
                ax[0].plot(f[key][:400], color =  'red',label=name)
            elif tot == 2:
                ax[0].plot(f[key][:400], color = 'b',label=name)
            elif tot == 3:
                ax[1].plot(f[key][:400], color =  'red',label=name)
            elif tot == 4:
                ax[1].plot(f[key][:400], color = 'b',label=name)
            print(f" size = {len(f[key])}", end=' ')
            # print(f"client = {i} best_acc")
            epochs = len(f[key])
            print('last_acc = {:.4f} '.format(f[key][epochs - 1]), end=' ')
            print('best_acc = {:.4f} '.format(max(f[key])))


    # ax[0].xlabel('训练轮数')
    # ax[0].ylabel('准确率')
    # ax[1].xlabel('训练轮数')
    # ax[1].ylabel('准确率')
    # plt.title('a')
    # plt.legend()
    # plt.title('mnist dataset dirichlet alpha=100')
    # plt.title(f'alpha = {alpha}')
    # plt.title('{dataset} CNN alpha = 0.5')
    plt.show()
    plt.savefig('test.png')
