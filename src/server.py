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
import random
class Server:
    '''
    local_model is the model architecture of each client;
    global_model is the model need to be aggregated;
    '''
    def __init__(self, device, global_models,clients, args, test_data_loader=''):
        self.device = device
        self.global_models = global_models
        self.args = args
        self.total_clients =  self.args.num_users
        # indexes set of clients
        self.indexes = [i for i in range(self.total_clients)]
        # 获取全局数据集
        # self.get_global_dataset(domains, args)
        # 生成用户
        self.test_data_loader = test_data_loader
        self.clients = clients
        # self.send_parameters()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.model_num = args.model_num
        for i in range(len(self.global_models)):
            self.global_models[i].to(device)
        self.pre_result = []
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def average_weights(self, w):
        '''
        :param w: weights of each client
        :return: the average of the weights
        '''
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            cnt = 1
            for client in range(1,len(w)):
                if key in w[client].keys():
                    w_avg[key] += w[client][key]
                    cnt += 1
            w_avg[key] = torch.div(w_avg[key],  cnt)
        return w_avg


    # def get_parameters(self):
    #     local_weights = []
    #     for client in range(self.args.num_users):
    #         local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
    #     return local_weights
    #
    # def send_parameters(self):
    #     w_avg = self.global_model.state_dict()
    #     if self.args.policy == 0:   # separate training
    #         return
    #     elif self.args.policy >=1 and self.args.policy <= 3: # collaborate train a global model
    #         for client in range(self.args.num_users):
    #             local_model = self.clients[client].local_model.state_dict()
    #             for key in local_model.keys():
    #                 local_model[key] = w_avg[key]
    #             self.clients[client].local_model.load_state_dict(local_model)
    #     elif self.args.policy == 4 and self.args.policy == 5:
    # #         分类器不共享
    #         for client in range(self.args.num_users):
    #             local_model = self.clients[client].local_model.state_dict()
    #             for key in local_model.keys():
    #                 if 'fc' not in key:
    #                     local_model[key] = w_avg[key]
    #             self.clients[client].local_model.load_state_dict(local_model)

    def aggregate1(self):
        if self.args.policy == 0: # Individual
            pass
        # elif self.args.policy == 1: # FedAVG
        #     local_weights = []
        #     for client in range(self.args.num_users):
        #         local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
        #     w_avg = self.average_weights(local_weights)
        #     self.global_model.load_state_dict(w_avg)
        #     for client in range(self.args.num_users):
        #         self.clients[client].local_model.load_state_dict(w_avg)
        #         合并K个模型
        else:
            for client in range(self.args.num_users):
                for i in range(self.model_num):
                    self.clients[client].local_models[i].load_state_dict(self.global_models[i].state_dict())
    def aggregate(self):
        if self.args.policy == 0: # Individual
            pass
        #         合并K个模型
        elif self.args.policy in list([2,4,5,6]):
            w_list = [[] for i in range(self.args.model_num)]
            for client in range(self.args.num_users):
                for i in range(self.model_num):
                    w_list[i].append(self.clients[client].local_models[i].state_dict())
            for i,ls in enumerate(w_list):
                    self.global_models[i].load_state_dict(self.average_weights(ls))
            for client in range(self.args.num_users):
                for i in range(self.model_num):
                    self.clients[client].local_models[i].load_state_dict(self.global_models[i].state_dict())
        elif self.args.policy == 3:
            w_list = [[] for i in range(self.args.model_num)]
            for client in range(self.args.num_users):
                for i in range(self.model_num):
                    w_list[i].append(self.clients[client].local_models[i].state_dict())
            all_weight = [] # 求所有参数的平均
            for i,ls in enumerate(w_list):
                self.global_models[i].load_state_dict(self.average_weights(ls))
                all_weight.append(self.average_weights(ls))
            all_weight = self.average_weights(all_weight)
            for i in range(self.model_num):
                state = self.global_models[i].state_dict()
                a = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var',
                     'bn.num_batches_tracked', 'layer1', 'layer2', 'layer3', ]

                if self.args.share_layer == 1:
                    fusion_layer = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var','bn.num_batches_tracked']
                elif self.args.share_layer == 2:
                    fusion_layer = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var',
                     'bn.num_batches_tracked', 'layer1']
                elif self.args.share_layer == 3:
                    fusion_layer = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var',
                                    'bn.num_batches_tracked', 'layer1', 'layer2']
                elif self.args.share_layer == 4:
                    fusion_layer = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var',
                                    'bn.num_batches_tracked', 'layer1', 'layer2','layer3']
                for key in all_weight.keys():
                    for fusion in fusion_layer:
                        if fusion in key:
                            state[key] = all_weight[key]
                            break
                self.global_models[i].load_state_dict(state)
            for client in range(self.args.num_users):
                for i in range(self.model_num):
                    self.clients[client].local_models[i].load_state_dict(self.global_models[i].state_dict())

        else:
            print('error policy')




    def train(self):

        self.global_test_losses = []
        self.global_test_accs = []
        self.local_test_accs = []
        self.local_test_losses = []
        self.test_losses = []
        self.test_accs = []
        # local_weights = []
        # 先同步一次
        self.aggregate1()
        # print('s')
        for epoch in tqdm(range(self.args.epochs),desc=self.args.name):
            print(f'Start Training round: {epoch}')

            # 模块1 训练
            for client in range(self.args.num_users):
                print('client = ',client)
                self.clients[client].train()
            # 模块2 聚合
            if epoch < 1000:
                self.aggregate()
            # 模块3 预测
            local_test_losses = []
            local_test_accs = []
            # test on each clients
            for model in range(len(self.global_models)):
                acc, loss = self.clients[model].inference()
                print('model = ',model,' acc = ',acc,' loss = ',loss)
                local_test_accs.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))

            # test on model
            # for model in range(len(self.global_models)):
            #     acc, loss = self. inference2(self.global_models[model])
            #     print('model = ',model,' acc = ',acc,' loss = ',loss)
            #     local_test_accs.append(copy.deepcopy(acc))
            #     local_test_losses.append(copy.deepcopy(loss))

            global_test_acc,global_test_loss = self.inference1()

            self.local_test_accs.append(local_test_accs)
            self.local_test_losses.append(local_test_losses)

            self.test_losses.append(sum(local_test_losses)/len(local_test_losses))
            self.test_accs.append(sum(local_test_accs)/len(local_test_accs))

            self.global_test_accs.append(global_test_acc)
            self.global_test_losses.append(global_test_loss)

            # print the training information in this epoch

            print(f'\nCommunication Round: {epoch}   Policy: {self.args.policy}')
            print(f'Avg testing Loss: {self.test_losses[-1]}')
            print(f'Avg test Accuracy: {self.test_accs[-1]}')
            print(f'global testing Loss: {self.global_test_losses[-1]}')
            print(f'global test Accuracy: {self.global_test_accs[-1]}')


            # if self.args.early_stop and len(test_losses)>100:
            #   if min(test_losses[0:-50]) < min(test_losses[-50:]):
            #       break
            if epoch%10 == 0:
                self.save_result()


        self.save_result()
        self.print_res()
        return
    # 在全局数据集上测试
    def inference1(self):
        for i in range(len(self.global_models)):
            self.global_models[i].eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs_list = []
                # outputs = self.global_models[0](images)
                for i in range(self.args.model_num):
                    out = self.global_models[i](images)
                    outputs_list.append(out)
                outputs = sum(outputs_list)/len(outputs_list)

                # outputs = self.global_model(images)
                batch_loss = self.criterion(outputs, labels.long())
                loss += batch_loss.item()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
                total += len(labels)

            accuracy = correct / total
            loss = loss / total
        return accuracy, loss

    def inference2(self,model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                # outputs = self.global_model(images)
                batch_loss = self.criterion(outputs, labels.long())
                loss += batch_loss.item()
                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
                total += len(labels)

            accuracy = correct / total
            loss = loss / total
        return accuracy, loss

    def print_res(self):
        print(f'Final Accuracy :{self.global_test_accs[-1]}')
        print(f'Best Accuracy:{max(self.global_test_accs)}')
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
                    f'_N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})'\
                    f'_P({self.args.policy})_lr({self.args.lr})_model({self.args.model})' \
                    f'_M{self.args.model_num}_MP{self.args.model_num_per_client}_name({self.args.name}).json'
        # print('json_name = ',json_name)
        with open(json_name,mode='w+') as f:
            result = {}
            result['test_losses'] = self.test_losses
            result['test_accs'] = self.test_accs
            result['local_test_losses'] = self.local_test_losses
            result['local_test_accs'] = self.local_test_accs
            result['global_test_losses'] = self.global_test_losses
            result['global_test_accs'] = self.global_test_accs
            json.dump(result,f)
        print(json_name)
    def save_model(self,state='before_finetune'):
        for client in range(self.args.num_users):
            # acc, loss = self.clients[domain][client].inference()
            torch.save(self.clients[client].local_model,'./save/'+state+'.model')
