import copy
from tqdm import tqdm
import torch
import numpy as np
# args: lr,momentum,local_ep
import torch.nn as nn
import torch.nn.functional as F

class Client:
    def __init__(self, device,  local_model, train_dataloader, test_dataloader, args):
        self.device = device
        self.local_model = local_model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

        self.local_model.to(self.device)

        # define optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'StepLR':
            self.optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.1)
        elif self.args.optimizer == 'MultiStepLR':
            self.optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [50,150], gamma=0.1)
        else:
            raise NotImplementedError

        # define Loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.local_model.train()
        epoch_loss = []

        # print(f'domain: {self.domain}')
        for iter in range(self.args.local_ep):
            print(f'local epoch: {iter}')
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                log_probs = self.local_model(images)
                loss = self.criterion(log_probs, labels.long())
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item()/len(labels))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
   
            if 'self.scheduler' in vars():
                self.scheduler.step()
            # print('lr = ',self.optimizer.param_groups[0]['lr'])
        # return sum(epoch_loss) / len(epoch_loss+1e-9)

    def inference(self):
        self.local_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs = self.local_model(images)
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
    def predict_pub_data(self,pub_datloader):
        # self.local_model.eval()
        with torch.no_grad():
            pub_predict = []
            for batch_idx, (images, labels) in enumerate(pub_datloader):
                # print(batch_idx*self.args.local_bs)
                if batch_idx * self.args.local_bs > 1000:
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs = self.local_model(images)
                pub_predict.append(outputs)
                # images,labels = images.to('cpu'), labels.to('cpu')
        return pub_predict
    def train_on_pub_data(self,pub_dataloader,index,pub_predict_list):
        self.local_model.train()
        for batch_idx, (images, labels) in enumerate(pub_dataloader):
            if batch_idx * self.args.local_bs > 1000:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.local_model(images)
            kl_loss = 0
            if self.args.col_policy == 1:
                avg = torch.zeros_like(pub_predict_list[0][batch_idx])
                for i in range(len(pub_predict_list)):
                    avg += pub_predict_list[i][batch_idx]
                kl_loss += self.loss_kl(F.log_softmax(outputs, dim=1),
                                        F.softmax(avg, dim=1))
            elif self.args.col_policy == 2:
                for  i in range(len(pub_predict_list)):
                    if i != index:
                        kl_loss += self.loss_kl(F.log_softmax(outputs, dim=1),
                                                F.softmax(pub_predict_list[i][batch_idx], dim=1))
                kl_loss /= len(pub_predict_list) - 1
            self.optimizer.zero_grad()
            kl_loss.backward()
            self.optimizer.step()
