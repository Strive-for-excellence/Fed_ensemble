import torch
from utils.options import args_parser
import numpy as np
import copy
from src.client import Client
from src.server import Server
from utils.data import DataSet,get_pub_data,build_dataset
from model import *
def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number : {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Policy: {args.policy}')


def get_model(args):
    client_model = []
    # global_model = resnet32(num_classes=args.num_classes)
    global_model = ResNet18(num_classes=args.num_classes)
    # mnist
    if args.dataset == 'mnist':
        global_model = DNN(1*28*28,10,1200,args)
        if args.model == "CNN":
            global_model = LeNet_mnist()
    elif args.dataset == 'cifar10':
        if args.policy == 1:
            global_model = LeNet_cifar10()
        elif args.policy == 2:
            global_model = LeNet_cifar10_for_ensemble2(drop_rate=args.drop_rate)
        elif args.policy == 3:
            global_model = LeNet_cifar10_for_ensemble2(drop_rate=args.drop_rate)
        elif args.policy == 4:
            global_model = LeNet_cifar10_for_ensemble2(drop_rate=args.drop_rate)
        elif args.policy == 5:
            global_model = LeNet_cifar10_for_ensemble2(drop_rate=args.drop_rate)
        else:
            print('error input ')
            exit(0)
    else:
        print('error input')
        exit(0)

    for i in range( args.num_users):
        local_model = copy.deepcopy(global_model)
        if args.policy == 3:

            for name,param in local_model.named_parameters():
                # print(name)
                if 'conv2' not in name:
                    continue
                not_required_grad = 'conv2.' + str(i)
                if not_required_grad in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        client_model.append(local_model)
    return global_model,client_model

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # local_model = ResNet18(10)
    # print(f'Model Structure: {local_model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #  prepare data

    # prepare model
    # global_model = ResNet101(args.num_classes)

    global_model,client_model = get_model(args)

    dataset = DataSet(args)
    clients = [Client(device, client_model[i], dataset.train[i], dataset.test[i], args) for i in range(args.num_users)]

    train_set,test_set = get_pub_data(args.pub_data,args)
    server = Server(device,global_model,clients,args,pub_data=train_set)
    server.train()
    # server.fine_fune()
    # server.train()
    server.print_res()
    server.save_result()


if __name__ == '__main__':


    args = args_parser()
    args.verbose = 0
    # set random seed
    np.random.seed(args.seed)
    exp_parameter(args)
    print(args)

    train(args)


