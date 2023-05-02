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
    alpha = 100
    dataset = 'cifar10'
# 0.63 0.68 0.05
# 0.59 0.65 0.06
# 0.54 0.60 0.06
    # server_path = r'\\192.168.210.100\mnt\fs1\openpai\zhangxingyan'
    Fed_AVGPath = r"\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_AVG\save"
    FedEnsPath = r"\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save"
    # print(os.path.exists(server_path+"\Fed_ensemblev2.2\save"))
 # “\\192.168.210.100\mnt\fs1\openpai\zhangxingyan\Fed_ensemblev2.2\save\Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name(cifar10_alpha_0.1_P_2_M1).json”
    file_name = {
        #                                          Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(0)_lr(0.01)_model(LeNet)_name(cifar10_alpha_d_100_P_0_D_1000).json
        # "Individual":  Fed_AVGPath+ f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(0)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_0_D_1000).json',
        "FecAVG":      Fed_AVGPath+ f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(1)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_1_D_1000).json',
        # "FedPer":      Fed_AVGPath+ f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_2_D_1000).json',
        # "FedProx":      Fed_AVGPath+ f'/Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(3)_lr(0.01)_model(LeNet)_name({dataset}_alpha_d_{alpha}_P_3_D_1000).json',

        # "FedEns1": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M1_MP1_name({dataset}_alpha_{alpha}_P_2_M1).json',
        "FedEns2": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(4)_lr(0.01)_model(LeNet)_M2_MP2_name({dataset}_alpha_{alpha}_P_4_M2).json',
        "FedEns3": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M3_MP3_name({dataset}_alpha_{alpha}_P_2_M3).json',
        "FedEns4": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M4_MP4_name({dataset}_alpha_{alpha}_P_2_M4).json',

        "FedEns5": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M5_MP5_name({dataset}_alpha_{alpha}_P_2_M5).json',
        "FedEns6": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M6_MP6_name({dataset}_alpha_{alpha}_P_2_M6).json',
        "FedEns7": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(2)_lr(0.01)_model(LeNet)_M7_MP7_name({dataset}_alpha_{alpha}_P_2_M7).json',
        # "FedEns14": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(4)_lr(0.01)_model(LeNet)_M1_MP1_name({dataset}_alpha_{alpha}_P_4_M1).json',
        # "FedEns24": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(4)_lr(0.01)_model(LeNet)_M2_MP2_name({dataset}_alpha_{alpha}_P_4_M2).json',
        # "FedEns54": FedEnsPath + f'\Result_dataset({dataset})_N(20)_E(5)_trainnum(500)_P(4)_lr(0.01)_model(LeNet)_M5_MP5_name({dataset}_alpha_{alpha}_P_4_M5).json',

    }
    # Result_dataset(cifar10)_N(20)_E(5)_trainnum(500)_P(2)_M3_MP3_name(cifar10_alpha_100_P_2).json


    for name,file in file_name.items():
        try:
            with open(file) as f:
                f  = json.load(f)
                key = 'global_test_accs'
                # key = 'test_accs'
                print(name)
                num = len(f[key])
                # for i in range(num):
                plt.plot(f[key][:500], label=name)

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
