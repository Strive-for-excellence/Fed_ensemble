import copy
import torch
# import numpy as np
# from utils.load_dataset import digits_dataset_read, digits_dataset_read_test,digits_dataset_loader, PairedData
from src.client import Client
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
import numpy as np
class Server:
    '''
    local_model is the model architecture of each client;
    global_model is the model need to be aggregated;
    '''
    def __init__(self, device, global_model,clients, args, pub_data=''):
        self.device = device
        self.global_model = global_model
        self.args = args
        self.total_clients =  self.args.num_users
        # indexes set of clients
        self.indexes = [i for i in range(self.total_clients)]
        # 获取全局数据集
        # self.get_global_dataset(domains, args)
        # 生成用户
        self.pub_data = pub_data
        self.clients = clients
        self.send_parameters()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

        self.pre_result = []

    def average_weights(self, w):
        '''
        :param w: weights of each client
        :return: the average of the weights
        '''
        w_avg = copy.deepcopy(w[0])

        for key in w_avg.keys():
            cnt = 1
            for client in range(1,self.args.num_users):
                if key in w[client].keys():
                    w_avg[key] += w[client][key]
                    cnt += 1
            w_avg[key] = torch.div(w_avg[key],  cnt)
            if (self.args.policy == 3 or self.args.policy == 4) and 'conv2' in key:
                pos1 = key.find('.')
                pos2 = key.find('.',pos1+1)
                idx = int(key[pos1+1:pos2])
                print('conv = {}, idx = {}'.format(key,idx))
                w_avg[key] = copy.deepcopy(w[idx][key])
        return w_avg

    def get_parameters(self):
        local_weights = []
        for client in range(self.args.num_users):
            local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
        return local_weights

    def send_parameters(self):
        w_avg = self.global_model.state_dict()
        if self.args.policy == 0:   # separate training
            return
        elif self.args.policy >=1 and self.args.policy <= 3: # collaborate train a global model
            for client in range(self.args.num_users):
                local_model = self.clients[client].local_model.state_dict()
                for key in local_model.keys():
                    local_model[key] = w_avg[key]
                self.clients[client].local_model.load_state_dict(local_model)
        elif self.args.policy == 4 and self.args.policy == 5:
    #         分类器不共享
            for client in range(self.args.num_users):
                local_model = self.clients[client].local_model.state_dict()
                for key in local_model.keys():
                    if 'fc' not in key:
                        local_model[key] = w_avg[key]
                self.clients[client].local_model.load_state_dict(local_model)

    def train(self):
        test_losses = []
        test_acc = []
        self.local_test_acc = []
        self.local_test_losses = []
        # local_weights = []
        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')

            # 模块1 训练
            for client in range(self.args.num_users):
                print('client = ',client)
                self.clients[client].train()
            # 模块2 聚合
            # # send parameters to each client
            weights = self.get_parameters()
            weight = self.average_weights(weights)
            self.global_model.load_state_dict(weight)
            self.send_parameters()
            # 模块3 预测
            local_test_losses = []
            local_test_acc = []
            # test on each clients
            for client in range(1):
                acc, loss = self.clients[client].inference()
                print('client = ',client,' acc = ',acc,' loss = ',loss)
                local_test_acc.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))

            test_losses.append(sum(local_test_losses)/len(local_test_losses))
            test_acc.append(sum(local_test_acc)/len(local_test_acc))



            # print the training information in this epoch

            print(f'\nCommunication Round: {epoch}   Policy: {self.args.policy}')
            print(f'Avg testing Loss: {test_losses[-1]}')
            print(f'Avg test Accuracy: {test_acc[-1]}')
            self.local_test_acc.append(local_test_acc)
            self.local_test_losses.append(local_test_losses)
            self.test_losses = test_losses
            self.test_acc = test_acc
            # if self.args.early_stop and len(test_losses)>100:
            #   if min(test_losses[0:-50]) < min(test_losses[-50:]):
            #       break
            if epoch%10 == 0:
                self.save_result()

        self.test_losses = test_losses
        self.test_acc = test_acc
        self.save_result()
        self.print_res()
        return

    def print_res(self):
        print(f'Final Accuracy :{self.test_acc[-1]}')
        print(f'Best Accuracy:{max(self.test_acc)}')
        # for domain in self.domains:
        #     print(f'domain: {domain}')
        #     print(f'Best accuracy: {max(self.domain_test_acc[domain])}')

    def save_result(self):
        # import shelve
        # from contextlib import closing
        # with closing(shelve.open(f'./save/Result_R({self.args.epochs})_'
        #     f'N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})_name({self.args.name}'
        #                          f'_P({self.args.policy})','c')) as shelve:
        #     shelve['test_losses'] = self.test_losses
        #     shelve['test_acc'] = self.test_acc
        import json
        json_name = f'./save/Result'\
                f'_dataset({self.args.dataset})'\
                f'_R({self.args.epochs})'\
                f'_N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})'\
                f'_P({self.args.policy})_lr({self.args.lr})_name({self.args.name}).json'
        # print('json_name = ',json_name)
        with open(json_name,mode='w+') as f:
            result = {}
            result['test_losses'] = self.test_losses
            result['test_acc'] = self.test_acc
            result['local_test_losses'] = self.local_test_losses
            result['local_test_acc'] = self.local_test_acc
            json.dump(result,f)
        print(json_name)
    def save_model(self,state='before_finetune'):
        for client in range(self.args.num_users):
            # acc, loss = self.clients[domain][client].inference()
            torch.save(self.clients[client].local_model,'./save/'+state+'.model')
