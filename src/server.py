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
    def __init__(self, device, global_model,clients, args, test_data_loader=''):
        self.device = device
        self.global_model = global_model
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
        self.global_model.to(device)
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
            for client in range(1,self.args.num_users):
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
    def aggregate(self):
        if self.args.policy == 0: # Individual
            pass
        elif self.args.policy == 1: # FedAVG
            local_weights = []
            for client in range(self.args.num_users):
                local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
            w_avg = self.average_weights(local_weights)
            self.global_model.load_state_dict(w_avg)
            for client in range(self.args.num_users):
                self.clients[client].local_model.load_state_dict(w_avg)
        elif self.args.policy == 2: # FedEnsemble
            local_weights = []
            for client in range(self.args.num_users):
                local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
            w_avg = self.average_weights(local_weights)
            # 需要ensemble的层
            for key in w_avg.keys():
                if 'res2.conv_bn_relu' in key and 'convs' in key:
                    pos1 = key.find('convs')
                    pos2 = key.find('.', pos1 + 6)
                    id = int(key[pos1+6:pos2])
                    w_avg[key] = local_weights[id][key]
            #  个性化的层：全连接层和最后一个block的BN层
            personalize_layer = ['fc','res2.conv_bn_relu_1.1','res2.conv_bn_relu_2.1','res2.conv_bn_relu_3.1']
            for client in range(0,self.args.num_users):
                for key in local_weights[client].keys():
                    per = False
                    for layer in personalize_layer:
                        if layer in key:
                            per = True
                    if not per:
                        local_weights[client][key] = w_avg[key]
                self.clients[client].local_model.load_state_dict(local_weights[client])

        elif self.args.policy == 3: # FedPer
            local_weights = []
            for client in range(self.args.num_users):
                local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
            w_avg = self.average_weights(local_weights)
            self.global_model.load_state_dict(w_avg)
            personalize_layer = ['fc']
            for client in range(0,self.args.num_users):
                for key in local_weights[client].keys():
                    per = False
                    for layer in personalize_layer:
                        if layer in key:
                            per = True
                    if not per:
                        # print(key)
                        local_weights[client][key] = w_avg[key]

                self.clients[client].local_model.load_state_dict(local_weights[client])
        elif self.args.policy == 4: # FenEns_global
            local_weights = []
            for client in range(self.args.num_users):
                local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
            w_avg = self.average_weights(local_weights)
            # 需要ensemble的层
            for key in w_avg.keys():
                if 'res2.conv_bn_relu' in key and 'convs' in key:
                    pos1 = key.find('convs')
                    pos2 = key.find('.', pos1 + 6)
                    id = int(key[pos1+6:pos2])
                    w_avg[key] = local_weights[id][key]
            #  使用全局模型
            for client in range(0,self.args.num_users):
                self.clients[client].local_model.load_state_dict(w_avg)
            global_model_state_dict = self.global_model.state_dict()
            self.global_model.load_state_dict(w_avg)
    # 在全局数据集上测试
    def inference(self):
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs = self.global_model(images)
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
    def train(self):

        self.global_test_losses = []
        self.global_test_accs = []
        self.local_test_accs = []
        self.local_test_losses = []
        self.test_losses = []
        self.test_accs = []
        # local_weights = []
        # 先同步一次
        self.aggregate()
        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')

            # 模块1 训练
            for client in range(self.args.num_users):
                print('client = ',client)
                self.clients[client].train()
            # 模块2 聚合
            self.aggregate()
            # 模块3 预测
            local_test_losses = []
            local_test_accs = []
            # test on each clients
            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()
                print('client = ',client,' acc = ',acc,' loss = ',loss)
                local_test_accs.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))
            policy_list = [1,4] # FedAVG 和 FedEns的全部聚合,在平衡数据集上测试
            if self.args.policy in policy_list:
                global_test_acc,global_test_loss = self.inference()
            else:
                global_test_acc, global_test_loss = 0.0,0.0
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

    def print_res(self):
        print(f'Final Accuracy :{self.test_accs[-1]}')
        print(f'Best Accuracy:{max(self.test_accs)}')
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
                f'_P({self.args.policy})_lr({self.args.lr})_name({self.args.name}).json'
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
