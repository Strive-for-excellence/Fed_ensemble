import copy
from tqdm import tqdm
import torch
import numpy as np
# args: lr,momentum,local_ep
import torch.nn as nn
import torch.nn.functional as F
from utils.get_model import Get_model
from utils import *
class Client:
    def __init__(self, device,  train_dataloader, test_dataloader, args):
        self.device = device
        # model = Get_model()
        self.local_models = [Get_model(args) for i in range(args.model_num_per_client)]

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        for i in range(args.model_num_per_client):
            self.local_models[i].to(self.device)
        self.model_num_per_client = args.model_num_per_client

        param_groups = []
        for i in range(args.model_num_per_client):
            param_groups  += [{"params": self.local_models[i].parameters(),"lr": args.lr}]

        self.Linear = torch.ones(1,self.args.model_num_per_client,requires_grad=True)
        # self.Linear = self.Linear.reshape((args.model_num_per_client,1,1))
        self.Linear = self.Linear.to(self.device)
        param_groups += [{"params":nn.Parameter(self.Linear),"lr":0.001}]
        # param_groups
        self.optimizers = torch.optim.SGD(param_groups, lr=self.args.lr, momentum=self.args.momentum)
        # if self.args.optimizer == 'sgd':
        #     self.optimizers = [torch.optim.SGD(self.local_models[i].parameters(), lr=self.args.lr, momentum=self.args.momentum) for i in range(self.model_num_per_client)]
        # self.model_idxs = [i for i in range(self.model_num_per_client)]
        # define Loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    def train(self):
        for i in range(self.model_num_per_client):
            self.local_models[i].train()
        epoch_loss = []

        # print(f'domain: {self.domain}')
        for iter in range(self.args.local_ep):
            print(f'local epoch: {iter}')
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                losses = []
                for i in range(self.model_num_per_client):

                    log_probs = self.local_models[i](images)
                    loss_i = self.criterion(log_probs, labels.long())
                    losses.append(loss_i.reshape(1))
                    # losses.append(loss_i)
                loss = torch.cat(losses).reshape(1,-1)
                if self.args.policy == 6:
                    self.optimizers.zero_grad()
                    for ls in losses:
                        ls.backward()
                    self.optimizers.step()
                    loss = torch.sum(loss)
                else:
                    if self.args.policy == 2 or self.args.policy == 3:
                        # self.Linear = self.Linear.data
                        loss = loss*self.Linear
                        loss = torch.sum(loss)
                    elif self.args.policy == 4:
                        loss = torch.sum(loss)
                    elif self.args.policy == 5:
                        loss = -torch.log(torch.sum(torch.exp(-loss)*self.Linear))
                    # loss =  torch.sum(loss)
                    self.optimizers.zero_grad()
                    loss.backward()
                    self.optimizers.step()
                batch_loss.append(loss.item() / len(labels))
                    # loss.backward()
                    # self.optimizers[i].step()
                    # batch_loss.append(loss.item()/len(labels))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
   
            # if 'self.scheduler' in vars():
            #     self.scheduler.step()
            # print('lr = ',self.optimizer.param_groups[0]['lr'])
        # return sum(epoch_loss) / len(epoch_loss+1e-9)

    def inference(self):
        for i in range(self.model_num_per_client):
            self.local_models[i].eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = []
                for i in range(self.model_num_per_client):
                    log_probs = self.local_models[i](images)
                    outputs.append(log_probs)
                outputs = torch.stack(outputs)
                tmp = copy.deepcopy(self.Linear.data).T
                tmp = tmp.resize(self.args.model_num_per_client,1,1)
                predict = outputs*tmp
                predict = torch.sum(predict,dim=0)
                batch_loss = self.criterion(predict, labels.long())
                loss += batch_loss.item()

                # prediction
                _, pred_labels = torch.max(predict, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
                total += len(labels)

            accuracy = correct / total
            loss = loss / total
        return accuracy, loss
    def predict_pub_data(self,pub_datloader):
        # self.local_model.eval()
        with torch.no_grad():
            pub_predict = []
            for batch_idx, (images, labels) in enumerate(pub_datloader):
                # print(batch_idx*self.args.local_bs)
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs = self.local_model(images)
                pub_predict.append(outputs)
                # images,labels = images.to('cpu'), labels.to('cpu')
        return pub_predict
    # def train_on_pub_data(self,pub_dataloader,index,pub_predict_list):
    #     self.local_model.train()
    #     for batch_idx, (images, labels) in enumerate(pub_dataloader):
    #         if batch_idx * self.args.local_bs > 1000:
    #             break
    #         images, labels = images.to(self.device), labels.to(self.device)
    #
    #         outputs = self.local_model(images)
    #         kl_loss = 0
    #         if self.args.col_policy == 1:
    #             avg = torch.zeros_like(pub_predict_list[0][batch_idx])
    #             for i in range(len(pub_predict_list)):
    #                 avg += pub_predict_list[i][batch_idx]
    #             kl_loss += self.loss_kl(F.log_softmax(outputs, dim=1),
    #                                     F.softmax(avg, dim=1))
    #         elif self.args.col_policy == 2:
    #             for  i in range(len(pub_predict_list)):
    #                 if i != index:
    #                     kl_loss += self.loss_kl(F.log_softmax(outputs, dim=1),
    #                                             F.softmax(pub_predict_list[i][batch_idx], dim=1))
    #             kl_loss /= len(pub_predict_list) - 1
    #         self.optimizer.zero_grad()
    #         kl_loss.backward()
    #         self.optimizer.step()
