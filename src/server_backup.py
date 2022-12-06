import copy
import torch
# import numpy as np
# from utils.load_dataset import digits_dataset_read, digits_dataset_read_test,digits_dataset_loader, PairedData
from src.client import Client
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

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

    # def get_model(self):
    #
    #     client_model = []
    #     self.global_model = ResNet101(self.args.class_nums)
    #     if self.args.policy == 3:
    #         half = self.args.num_users//2
    #         for i in range(half):
    #             client_model.append(copy.deepcopy(self.global_model))
    #         for i in range(half,self.args.num_users):
    #             client_model.append(ResNet(Bottleneck, [3, 4, 22, 3], num_classes=self.args.num_classes))
    #
    #     else :
    #         for i in range(self.args.num_users):
    #             client_model.append(copy.deepcopy(self.global_model))
    #
    #     return client_model
    #     # for i in range(self.args.num_users):
    #     # self.send_parameters(model.state_dict())/
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
        return w_avg

    def get_parameters(self):
        local_weights = []
        for client in range(self.args.num_users):
            local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
        return local_weights

    def send_parameters(self):
        w_avg = self.global_model.state_dict()
        if self.args.policy == 1:   # separate training
            return
        elif self.args.policy >= 2: # collaborate train a global model
            for client in range(self.args.num_users):
                local_model = self.clients[client].local_model.state_dict()
                for key in local_model.keys():
                    local_model[key] = w_avg[key]
                self.clients[client].local_model.load_state_dict(local_model)
            return
    def col_train(self):
        for batch_idx, (images, labels) in enumerate(self.pub_data):
            # if batch_idx * self.args.local_bs > 1000:
            #     break
            outputs = []
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                for i in range(self.args.num_users):
                    outputs.append(self.clients[i].local_model(images))
            for i in range(self.args.num_users):
                self.clients[i].local_model.train()
                predict = self.clients[i].local_model(images)
                kl_loss = 0
                if self.args.col_policy == 1:
                    for j in range(self.args.num_users):
                            kl_loss = 0
                            for j in range(self.args.num_users):
                                if i != j:
                                    kl_loss += self.loss_kl(F.log_softmax(predict, dim=1),
                                                            F.softmax(outputs[j], dim=1))
                            kl_loss /= self.args.num_users - 1
                else:
                    kl_loss = 0
                    avg = torch.zeros_like(outputs[i])
                    avg.requires_grad = False
                    for j in range(self.args.num_users):
                        if i != j:
                            avg += outputs[j]
                    avg /= self.args.num_users-1
                    kl_loss = self.loss_kl(F.log_softmax(predict,dim=1),
                                           F.softmax(avg, dim=1))
                if self.args.pub_data_labeled:
                    ce_loss = self.loss_ce(outputs[i],labels)
                    kl_loss += ce_loss
                self.clients[i].optimizer.zero_grad()
                kl_loss.backward()
                self.clients[i].optimizer.step()
    def train(self):
        train_losses = []
        test_losses = []
        test_acc = []

        # local_weights = []
        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')
            # train_losses.clear()
            # test_losses.clear()
            # test_acc.clear()
            local_train_losses = []
            local_test_losses = []
            local_test_acc = []


            # select clients to train their local model
            # 获得模型对公共数据集的预测
            if self.args.col_policy != 0:
                # pub_predict_list = []
                # for client in range(self.args.num_users):
                #     pub_predict_list.append(self.clients[client].predict_pub_data(self.pub_data))
                # for client in range(self.args.num_users):
                #     self.clients[client].train_on_pub_data(self.pub_data,client,pub_predict_list)
                for e in range(self.args.col_epoch):
                    self.col_train()
            # test on each clients

            for client in range(self.args.num_users):
                print('client = ',client)
                loss = self.clients[client].train()
                # local_train_losses.append(loss)
            local_train_losses_avg = sum(local_train_losses) / len(local_train_losses)
            train_losses.append(local_train_losses_avg)


            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()
                print('client = ',client,' acc = ',acc,' loss = ',loss)
                local_test_acc.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))
            test_losses.append(sum(local_test_losses)/len(local_test_losses))
            test_acc.append(sum(local_test_acc)/len(local_test_acc))

            # clients send parameters to the server
            local_weights = self.get_parameters()
            w_avg = self.average_weights(local_weights)
            self.global_model.load_state_dict(w_avg)

            # send parameters to each client
            self.send_parameters()


            # # send parameters to each client
            # self.send_parameters(w_avg)

            # print the training information in this epoch
            print(f'\nCommunication Round: {epoch}   Policy: {self.args.policy}')
            print(f'Avg training Loss: {train_losses[-1]}')
            print(f'Avg testing Loss: {test_losses[-1]}')
            print(f'Avg test Accuracy: {test_acc[-1]}')
            self.train_losses = train_losses
            self.test_losses = test_losses
            self.test_acc = test_acc
            # if self.args.early_stop and len(test_losses)>100:
            #   if min(test_losses[0:-50]) < min(test_losses[-50:]):
            #       break
            if epoch%10 == 0:
                self.save_result()

        self.train_losses = train_losses
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
        #     shelve['train_losses'] = self.train_losses
        #     shelve['test_losses'] = self.test_losses
        #     shelve['test_acc'] = self.test_acc
        import json
        with open(f'./save/Result'
                f'_dataset({self.args.dataset})'
                f'_R({self.args.epochs})'
                f'_N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})'
                f'_P({self.args.policy})_name({self.args.name}).json',mode='w+') as f:
            result = {}
            result['train_losses'] = self.train_losses
            result['test_losses'] = self.test_losses
            result['test_acc'] = self.test_acc
            json.dump(result,f)
        print('name = '
                f'./save/Result'
                f'_dataset({self.args.dataset})'
                f'_R({self.args.epochs})'
                f'_N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})'
                f'_P({self.args.policy})_name({self.args.name}).json')
    def save_model(self,state='before_finetune'):
        for client in range(self.args.num_users):
            # acc, loss = self.clients[domain][client].inference()
            torch.save(self.clients[client].local_model,'./save/'+state+'.model')
