#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import time
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=1, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    # parser.add_argument('--local_data_num', type=int, default=10000000, help='image size after loading')

    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--step_size',type=int,default='30',help='step size  of StepLR')
    # dataset argument
    parser.add_argument('--data_dir', type=str, default='./data/Digit-Five/')
    parser.add_argument('--train_num', type=int, default=500, help='number of training samples for training')
    parser.add_argument('--test_num', type=int, default=100, help='number of testing samples for training')
    parser.add_argument('--scale', type=int, default=32, help='image size after loading')
    # parser.add_argument('--class_nums', type=int, default=10, help='class number')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--use_all_data',type= int, default=0, help='1 every client use all data')
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--noniid', type=str, default='dirichlet',
                        help='Default set to pathological Non-IID.')
    # parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.1, help='the degree of imbalance')

    # model arguments
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")


    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')


    parser.add_argument('--name',type=str, default='time',help='name of process')
    # mixture global and local representation arguments
    parser.add_argument('--policy', type=int, default=1, choices=[0,1,2,3,4,5], help='global training policy')
    # parser.add_argument('--domains', type=str, default='mnistm,mnist,usps,syn',
    #                     help='different domain in federated learning')
    parser.add_argument('--early_stop',type=int,default=1,help='early_stop')
    parser.add_argument('--use_small',type=int,default=1,help='use_small')
    parser.add_argument('--net_depth',type=int,default=-1,help='net_depth')
    # col_train
    parser.add_argument('--col_policy',type=int,default=0,help='col policy')
    parser.add_argument('--col_epoch',type=int, default=2,help='col train epoch')
    parser.add_argument('--pub_data',type=str, default='', help = 'public dataset')
    parser.add_argument('--pub_data_num',type=int,default='5000',help='public_dataset_size')
    parser.add_argument('--pub_data_labeled',type=int,default=0,help='0 unlabel,1 label')
    parser.add_argument('--use_pseudo_label',type=int,default=0,help='0 not use_pseudo_label,1 use_pseudo_label')
    parser.add_argument('--use_avg_loss',type=int,default=0,help='0 not use ,1 use avg loss, 2 use weight avg loss')
    parser.add_argument('--weight_temperature',type=float,default=1,help='temperature of weight caculation ')
    parser.add_argument('--ema',type=float,default=0,help='Use or not use ema label')
    parser.add_argument('--kalman',type=int,default=0,help='Use or not use kalman ')
    parser.add_argument('--forward_times',type=int,default=5,help='MC dropout times')
    parser.add_argument('--col_threshold',type=int,default=0,help='col policy threshold epoch')
    parser.add_argument('--drop_rate',type=float,default=0,help='dropout rate')
    args = parser.parse_args()
    if args.name == 'time':
        args.name = str(time.localtime().tm_year)+str(time.localtime().tm_mon)+str(time.localtime().tm_mday)+str(time.localtime().tm_hour)+str(time.localtime().tm_min)+str(time.localtime().tm_sec)
    if args.pub_data == "":
        args.pub_data = args.dataset
    return args
