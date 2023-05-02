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
    def __init__(self, device, global_models,clients, args, test_data_loader='',pub_data=''):
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
        self.pub_data = pub_data
        self.clients = clients
        # self.send_parameters()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.model_num = args.model_num
        for i in range(len(self.global_models)):
            self.global_models[i].to(device)
        self.pre_result = []
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizers = [torch.optim.SGD(global_models[i].parameters(), lr=self.args.lr, momentum=self.args.momentum) for i in range(len(global_models))]
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
                for i, idx in enumerate(self.clients[client].model_idxs):
                    self.clients[client].local_models[i].load_state_dict(self.global_models[idx].state_dict())
    def aggregate(self):
        if self.args.policy == 0: # Individual
            pass
        #         合并K个模型
        # elif self.args.policy in list([2,4,5,6]):
        #     w_list = [[] for i in range(self.args.model_num)]
        #     for client in range(self.args.num_users):
        #         for i in range(self.model_num):
        #             w_list[i].append(self.clients[client].local_models[i].state_dict())
        #     for i,ls in enumerate(w_list):
        #             self.global_models[i].load_state_dict(self.average_weights(ls))
        #     for client in range(self.args.num_users):
        #         for i in range(self.model_num):
        #             self.clients[client].local_models[i].load_state_dict(self.global_models[i].state_dict())
        # elif
        elif self.args.policy == 3 or self.args.policy == 4:
            w_list = [[] for i in range(self.args.model_num)]
            # all_weight = []
            for client in range(self.args.num_users):
                for i, idx in enumerate(self.clients[client].model_idxs):
                    w_list[idx].append(self.clients[client].local_models[i].state_dict())
                    # all_weight.append(self.clients[client].local_models[i].state_dict())
            for i, ls in enumerate(w_list):
                if ls:
                    self.global_models[i].load_state_dict(self.average_weights(ls))
            if self.args.policy == 4:
                self.col_train()
            for client in range(self.args.num_users):
                for i, idx in enumerate(self.clients[client].model_idxs):
                    self.clients[client].local_models[i].load_state_dict(self.global_models[idx].state_dict())

        else:
            print('error policy')

    # use_avg_loss
    # 1 平均
    # 2 加权
    # 其它忽略
    def col_train(self):
        # with tqdm(total=self.args.pub_data_num) as pbar:
        if 1:
            for batch_idx, (images, labels) in enumerate(self.pub_data):

                data_num = images.shape[0]
                outputs = []
                images, labels = images.to(self.device), labels.to(self.device)
                images, labels = Variable(images), Variable(labels)
                # with torch.no_grad():

                #    use_avg_loss = 1
                # FedMd
                if self.args.use_avg_loss == 1:
                    with torch.no_grad():
                        # model.eval()
                        for client in range(self.args.num_users):
                            self.clients[client].local_model.eval()
                            outputs.append(F.softmax(self.clients[client].local_model(images), dim=1))
                    avg_soft_label = Variable(sum(outputs) / len(outputs))
                #
                elif self.args.use_avg_loss == 2 or \
                        self.args.use_avg_loss == 4 or \
                        self.args.use_avg_loss == 5:
                    # 2,5 只需要预测一次
                    if self.args.use_avg_loss == 2 or \
                            self.args.use_avg_loss == 5:
                        with torch.no_grad():
                            for client in range(self.args.num_users):
                                # self.clients[client].local_model.eval()
                                outputs.append(F.softmax(self.clients[client].predict_data(images), dim=1))
                        outputs_tmp = [outputs[i].cpu().numpy() for i in
                                       range(self.args.num_users)]
                    # # 4 蒙特卡洛需要预测多次
                    # elif self.args.use_avg_loss == 4:
                    #     client_item_mean = []
                    #     with torch.no_grad():
                    #         for client in range(self.args.num_users):
                    #             self.clients[client].local_model.train()
                    #             # self.clients[client].local_model.eval()
                    #             times = self.args.forward_times
                    #             results = []
                    #             for time in range(times):
                    #                 result = F.softmax(self.clients[client].local_model(images), dim=1)
                    #                 result = result.detach().cpu().numpy()
                    #                 results.append(result)
                    #             item_mean = np.mean(results, axis=0)
                    #             client_item_mean.append(item_mean)
                    #
                    #     outputs = client_item_mean
                    #     outputs = torch.tensor(outputs)
                    #     outputs_tmp = np.array(outputs)

                    outputs_entropy = []
                    # 2,4 使用预测熵
                    if self.args.use_avg_loss == 2 or self.args.use_avg_loss == 4:
                        if self.args.kalman == 0:
                            for i in range(self.args.num_users):
                                outputs_entropy.append(
                                    - np.sum(outputs_tmp[i] * np.log(outputs_tmp[i]),
                                             axis=-1))
                            all_entropy = np.stack(outputs_entropy, axis=0)
                            all_entropy = torch.tensor(all_entropy) / self.args.weight_temperature
                            all_entropy = all_entropy / torch.sum(all_entropy, dim=0)
                            # all_entropy = F.softmax(all_entropy,dim=0)
                            all_entropy = torch.unsqueeze(all_entropy, -1)
                            # if batch_idx == 1:
                            #     print(all_entropy)
                            avg_soft_label = np.sum(np.array(outputs_tmp) * all_entropy.numpy(), axis=0)
                            avg_soft_label = torch.tensor(avg_soft_label)
                        else:
                            # kalman = 1
                            for i in range(self.args.num_users):
                                outputs_entropy.append(np.sum(-outputs_tmp[i] * np.log2(outputs_tmp[i]), axis=1))
                            all_entropy = np.stack(outputs_entropy, axis=0)
                            sigma_divided_by_1 = 1 / torch.tensor(np.square(all_entropy))
                            sum_sigma = 1 / torch.sum(sigma_divided_by_1, dim=0)
                            weight = sum_sigma * torch.tensor(sigma_divided_by_1)
                            weight = torch.unsqueeze(weight, -1)
                            avg_soft_label = torch.sum(torch.tensor(outputs_tmp) * weight, dim=0)
                    # 5 使用置信度
                    elif self.args.use_avg_loss == 5:
                        outputs_tmp = np.array(outputs_tmp)
                        weight = np.max(outputs_tmp, axis=2)
                        weight = weight / np.sum(weight, axis=0)
                        weight = torch.unsqueeze(torch.tensor(weight), -1)
                        avg_soft_label = torch.sum(torch.tensor(outputs_tmp) * weight, dim=0)
                    avg_soft_label = avg_soft_label.to(self.device)

                avg_soft_label = avg_soft_label.to(self.device)
                # 利用无标签数据集训练
                for i in range(self.args.model_num):
                    # predict = outputs[i]
                    self.global_models[i].train()
                    predict = self.global_models[i](images)
                    loss = self.loss_kl(F.log_softmax(predict, dim=1), avg_soft_label)
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

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
            # if epoch < 1000:
            self.aggregate()
            # 模块3 预测
            local_test_losses = []
            local_test_accs = []
            model_test_losses = []
            model_test_accs = []
            # test on each clients
            # 客户端test
            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()
                print('cient = ',client,' acc = ',acc,' loss = ',loss)
                local_test_accs.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))
            # 模型test
            for model in range(self.args.model_num):
                acc, loss =  self.inference2(self.global_models[model])
                print('model = ',model,' acc = ',acc,' loss = ',loss)
                model_test_accs.append(copy.deepcopy(acc))
                model_test_losses.append(copy.deepcopy(loss))

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
