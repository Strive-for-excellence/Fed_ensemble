import torch
import torchvision
import torchvision.transforms as transforms
import os
from utils.sampling import dirichlet_noniid, pathological_noniid
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
def build_dataset(args):
    print(f'==> Preparing data {args.dataset}')
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar10'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                               transform=transform_test)
    elif args.dataset =='cifar100':
        data_dir = '../data/cifar100'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                          transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                         transform=transform_test)
    elif args.dataset == 'tiny_imagenet':
        data_dir = '../data/tiny-imagenet-200'
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                          transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir,'val'),
                                         transform=transform_test)
    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        testset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    else:
        print('wrong dataset name')
    return trainset, testset

def get_pub_data(dataset_name,args):
    print(f'==> Preparing data {dataset_name}')
    if dataset_name == 'cifar10':
        # args.num_classes = 10
        data_dir = '../data/cifar10'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                               transform=transform_test)
    elif dataset_name =='cifar100':
        # args.num_classes = 100
        data_dir = '../data/cifar100'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                          transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                         transform=transform_test)
    elif dataset_name == 'tiny_imagenet':
        # args.num_classes = 200
        data_dir = '../data/tiny-imagenet-200'
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                          transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir,'val'),
                                         transform=transform_test)
    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        testset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    else:
        print('wrong dataset name')
    idxs = random.sample([_ for _ in range(len(trainset.targets))],min(args.pub_data_num,len(trainset.targets)))

    trainset = DatasetSplit(trainset,idxs)
    trainset = torch.utils.data.DataLoader(trainset,batch_size=args.local_bs,shuffle=False)
    testset = torch.utils.data.DataLoader(testset,batch_size=args.local_bs,shuffle=False)
    return trainset, testset

class DataSet:
    def __init__(self, args):
        trainset, testset = build_dataset(args)
        client_num = args.num_users
        batch_size = args.local_bs
        args.train_num = min(len(trainset) // client_num,args.train_num)
        ls = [args.train_num for _ in range(client_num)]
        ls.append(len(trainset)-sum(ls))
        if args.iid:
            data_list = torch.utils.data.random_split(trainset, ls)[:-1]
            self.train = [torch.utils.data.DataLoader(data_list[i], batch_size=batch_size, shuffle=True)  for i in range(client_num)]
            self.test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        else:
            if args.noniid == 'pathological':
                train_user_groups, test_user_groups = pathological_noniid(trainset, testset, args.num_users,
                                                                          args.alpha, args.seed, args)
            else:
                train_user_groups, test_user_groups = dirichlet_noniid(trainset, testset, args.num_users,
                                                                       args.alpha, args.seed, args)
            self.train = [torch.utils.data.DataLoader(DatasetSplit(trainset, train_user_groups[i]),
                                                      batch_size=batch_size, shuffle=True) for i in range(client_num)]
            self.test =  [torch.utils.data.DataLoader(DatasetSplit(testset, test_user_groups[i]),
                                batch_size=batch_size, shuffle=False) for i in range(client_num)]
            # self.test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        print('Data point each client = ',args.train_num)
        # del data_list

# print(len(dataset.train[0]))
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
