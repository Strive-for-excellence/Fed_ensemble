import torch
from utils.options import args_parser
import numpy as np
import copy
from src.client import Client
from src.server import Server
from utils.data import DataSet,get_pub_data,build_dataset
from utils.get_model import Get_model
from model import *
def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number : {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Policy: {args.policy}')

def get_model_set(args):
    model_set = [Get_model(args) for i in range(args.model_num)]
    return model_set
def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # local_model = ResNet18(10)
    # print(f'Model Structure: {local_model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #  prepare data

    # prepare model
    # global_model = ResNet101(args.num_classes)

    global_models = get_model_set(args)
    # print(Get_model(args))

    dataset = DataSet(args)
    clients = [Client(device,  dataset.train[i], dataset.test[i], args) for i in range(args.num_users)]

    train_set,test_set = get_pub_data(args.pub_data,args)
    server = Server(device,global_models,clients,args,test_data_loader=test_set)
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


