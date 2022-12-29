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

    file_name = {
        # f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_alpha_d_0.01_P_1).json',
        "Individual":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_N(4)_E(1)_trainnum(1000)_P(0)_lr(0.01)_name(cifar10_alpha_d_0.01_P_0).json',
        "FedAVG":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_alpha_d_0.01_P_1).json',
        "FedPer":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_N(4)_E(1)_trainnum(1000)_P(3)_lr(0.01)_name(cifar10_alpha_d_0.01_P_3).json',
        "FedEns":
            f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(cifar10)_N(4)_E(1)_trainnum(1000)_P(2)_lr(0.01)_name(cifar10_alpha_d_0.01_P_2).json',
    }

    # file_name = {
    #     "Individual":
    #         f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(mnist)_N(4)_E(1)_trainnum(1000)_P(0)_lr(0.01)_name(mnist_alpha_d_0.01_P_0).json',
    #     "FedAVG":
    #         f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(mnist)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(mnist_alpha_d_0.01_P_1).json',
    #     "FedPer":
    #         f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(mnist)_N(4)_E(1)_trainnum(1000)_P(3)_lr(0.01)_name(mnist_alpha_d_0.01_P_3).json',
    #     "FedEns":
    #         f'Z:\zhangxingyan\Fed_ensemble\save/Result_dataset(mnist)_N(4)_E(1)_trainnum(1000)_P(2)_lr(0.01)_name(mnist_alpha_d_0.01_P_2).json',
    # }
    # name = ["Ensebmle model + fedavg","Ensemble model + personalize conv","Ensemble model + personalize conv and classifier","Ensemble model + personalize classifier"]
    # name = ["Fedavg","Personalize Classifier","Ensemble model + personalize conv and classifier","Ensemble model + personalize classifier"]
    # name = ['dnn','cnn']
#     name = ['1 individual','2 avg','3 weight_avg','4 weight_avg_kalman','5 weight_avg_MCdropout','6 weight_avg_kalman_MCdropout','7 weight_avg_kalman_MCdropout_10','8','9']


    for name,file in file_name.items():

        with open(file) as f:
            f  = json.load(f)
            num = len(f['test_acc'])
            # for i in range(num):
            plt.plot(f['test_acc'][:40], label=name)

            print(f" size = {len(f['test_acc'])}",end=' ')
                # print(f"client = {i} best_acc")
            epochs = len(f['test_acc'])
            print('last_acc = {:.3f} '.format(f['test_acc'][epochs-1]),end = ' ')
            print('best_acc = {:.3f} '.format(max(f['test_acc'])))

            # print(f['test_acc'][200])
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
    # plt.title('mnist dataset dirichlet alpha=100')
    plt.title('alpha = 100')
    # plt.title('cifar10 CNN alpha = 0.5')
    # plt.show()
    plt.savefig('test.png')
