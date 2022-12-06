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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    file_name = [
        f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_alpha_d_0.1_P_1).json',
        f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(2)_lr(0.01)_name(cifar10_alpha_d_0.1_P_2).json',
        f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(3)_lr(0.01)_name(cifar10_alpha_d_0.1_P_3).json',
        f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(4)_lr(0.01)_name(cifar10_alpha_d_0.1_P_4).json',

    ]
    name = ["fedavg","Ensebmle model + fedavg","Ensemble model + personalize conv","Ensemble model + personalize conv and classifier"]
    # name = ['dnn','cnn']
#     name = ['1 individual','2 avg','3 weight_avg','4 weight_avg_kalman','5 weight_avg_MCdropout','6 weight_avg_kalman_MCdropout','7 weight_avg_kalman_MCdropout_10','8','9']
    for  i in range(10,100):
        name.append(str(i))

    for i in range(0,len(file_name)):
        with open(file_name[i]) as f:
            f  = json.load(f)
            num = len(f['test_acc'])
            # for i in range(num):
            plt.plot(f['test_acc'][:], label=name[i])

            print(f"idx = {i} size = {len(f['test_acc'])}",end=' ')
                # print(f"client = {i} best_acc")
            epochs = len(f['test_acc'])
            print('last_acc = {:.3f} '.format(f['test_acc'][epochs-1]),end = ' ')
            print('best_acc = {:.3f} '.format(max(f['test_acc'])))

            # print(f['test_acc'][200])
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
    # plt.title('mnist dataset dirichlet alpha=100')
    plt.title('alpha = 0.1')
    # plt.title('cifar10 CNN alpha = 0.1')
    # plt.show()
    plt.savefig('test.png')
