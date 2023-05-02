#!/bin/bash
#cifar100 实验环境中，share bottom架构，1
_dataset=$1
_policy=$2
_alpha=$3
_gpu=${4:-0}
_lr=${5:-0.01}
_model_num=${6:-1}
_share_layer=${7:-1}
echo python3 main.py  --dataset ${_dataset} --num_classes 10 --epochs 2000  --num_users 20 --local_ep 5 --local_bs 100 --train_num 500 --test_num 100 --lr ${_lr}  \
        --policy ${_policy}   \
        --iid 0 --noniid dirichlet --alpha ${_alpha}  \
        --model_num ${_model_num} --model_num_per_client ${_model_num} \
        --global_test_data_num 1000 \
        --gpu ${_gpu} \
        --share_layer ${_share_layer}
        --name ${_dataset}_alpha_${_alpha}_P_${_policy}_M${_model_num}_S${_share_layer}
python3 main.py  --dataset ${_dataset} --num_classes 10 --epochs 2000  --num_users 20 --local_ep 5 --local_bs 100 --train_num 500 --test_num 100 --lr ${_lr}  \
        --policy ${_policy}   \
        --iid 0 --noniid dirichlet --alpha ${_alpha}  \
        --model_num ${_model_num} --model_num_per_client ${_model_num} \
        --global_test_data_num 1000 \
        --gpu ${_gpu} \
        --share_layer ${_share_layer} \
        --name ${_dataset}_alpha_${_alpha}_P_${_policy}_M${_model_num}_S${_share_layer}
#
#bash run.sh cifar10 2 100 0 0.01 5 1&
#bash run.sh cifar10 2 100 0 0.01 2 1&
#bash run.sh cifar10 2 100 0 0.01 1 1&
#bash run.sh cifar10 2 1 1 0.01 5 1&
#bash run.sh cifar10 2 1 1 0.01 2 1&
#bash run.sh cifar10 2 1 1 0.01 1 1&
#bash run.sh cifar10 2 0.1 2 0.01 5 1&
#bash run.sh cifar10 2 0.1 2 0.01 2 1&
#bash run.sh cifar10 2 0.1 2 0.01 1 1