import torch
from utils.options import args_parser
import numpy as np
import copy
from src.client import Client
from src.server import Server
from utils.data import DataSet,get_pub_data,build_dataset
from utils.get_model import Get_model
# from torchvision.models import vgg19,alexnet

from model import *
def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number : {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Policy: {args.policy}')

def get_model_set(args):

    model_set = []
    # 模型数量，模型种类*每个客户端模型数量
    assert args.model_num == args.model_type_num * args.model_num_per_client
    for model_type in range(args.model_type_num):
        for i in range(args.model_num_per_client):
            if model_type == 0:
                model = LeNet(num_classes=args.num_classes)
            elif model_type == 1:
                model = ResNet8(num_classes=args.num_classes)
            elif model_type == 2:
                model = alexnet(num_classes=args.num_classes)
            elif model_type == 3:
                model = vgg19(num_classes=args.num_classes)

            model_set.append(copy.deepcopy(model))
    local_models = []
    idxs_list = []
    cluster_size = int((args.num_users-1)//args.model_type_num+1)
    for i in range(args.num_users):
        t = int(i // cluster_size) # 属于第几个类
        idxs_list.append([i for i in range(t*args.model_num_per_client,(t+1)*args.model_num_per_client)])
    client_model_list = []
    for i in range(args.num_users):
        client_model = [model_set[c] for c in idxs_list[i]]
        client_model_list.append(client_model)
    return model_set,client_model_list,idxs_list

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # local_model = ResNet18(10)
    # print(f'Model Structure: {local_model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #  prepare data

    # prepare model
    # global_model = ResNet101(args.num_classes)

    global_models,client_models,client_idxs = get_model_set(args)
    # print(Get_model(args))

    dataset = DataSet(args)
    clients = [Client(device,  dataset.train[i], dataset.test[i], args,client_models[i],client_idxs[i]) for i in range(args.num_users)]

    train_set,test_set = get_pub_data(args.pub_data,args)
    server = Server(device,global_models,clients,args,test_data_loader=test_set,pub_data=train_set)
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
    # model = Get_model(args)
    # print(model)
    # state_dict = model.state_dict()
    # print(state_dict.keys())
    # print(args)
    #
    train(args)


