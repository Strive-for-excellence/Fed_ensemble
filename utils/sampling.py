import numpy as np
from torchvision import datasets, transforms
import random
# alpha 不平衡程度，越小，不平衡程度越大，
def dirichlet_noniid(train_dataset, test_dataset, num_users, alpha, seed, args):
    num_train_indices = len(train_dataset.targets)  # the number of training samples
    num_test_indices = len(test_dataset.targets)    # the number of testing samples

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    num_class = len(np.unique(train_dataset.targets))
    train_idxs_classes = [[] for _ in range(num_class)]
    test_idxs_classes = [[] for _ in range(num_class)]

    # number of training and testing sample for each client

    num_client_train_sample = args.train_num
    num_client_test_sample = args.test_num
    # num_client_train_sample = int(num_train_indices / num_users)
    # num_client_test_sample = int(num_test_indices / num_users)
    # sorted by label, samples with the same labels are put in a list
    for i in range(num_train_indices):
        train_idxs_classes[train_labels[i]].append(i)
    for i in range(num_test_indices):
        test_idxs_classes[test_labels[i]].append(i)

    train_dict_users = {i: [] for i in range(num_users)}
    test_dict_users = {i: [] for i in range(num_users)}

    # generate the data distribution for each client
    # use random seed to initialize
    random_state = np.random.RandomState(seed)
    q = random_state.dirichlet(np.repeat(alpha, num_class), num_users)

    # partition dataset according to q for each client
    for i in range(num_users):
        # make sure that each client have num_client_train_sample samples
        temp_train_sample = num_client_train_sample
        temp_test_sample = num_client_test_sample
        # partition each class for clients
        for j in range(num_class):
            num_train = int(num_client_train_sample * q[i][j] + 0.5)
            num_test = int(num_client_test_sample * q[i][j] + 0.5)
            num_train = num_train if temp_train_sample - num_train >= 0 else temp_train_sample
            num_test = num_test if temp_test_sample - num_test >= 0 else temp_test_sample
            temp_train_sample -= num_train
            temp_test_sample -= num_test
            assert num_train >= 0 and num_test >= 0
            train_dict_users[i] += random_state.choice(train_idxs_classes[j], num_train, replace=False).tolist()
            test_dict_users[i] += random_state.choice(test_idxs_classes[j], num_test, replace=False).tolist()
        train_dict_users[i] = np.array(train_dict_users[i])
        test_dict_users[i] = np.array(test_dict_users[i])
    return train_dict_users, test_dict_users

def pathological_noniid(train_dataset, test_dataset, num_users, alpha, seed, args):
    assert(alpha>=1)
    alpha = int(alpha)
    num_train_indices = len(train_dataset.targets)  # the number of training samples
    num_test_indices = len(test_dataset.targets)  # the number of testing samples

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    num_class = len(np.unique(train_dataset.targets))
    train_idxs_classes = [[] for _ in range(num_class)]
    test_idxs_classes = [[] for _ in range(num_class)]

    num_client_train_sample = args.train_num
    num_client_test_sample = args.test_num

    # sorted by label, samples with the same labels are put in a list
    for i in range(num_train_indices):
        train_idxs_classes[train_labels[i]].append(i)
    for i in range(num_test_indices):
        test_idxs_classes[test_labels[i]].append(i)

    train_dict_users = {i: [] for i in range(num_users)}
    test_dict_users = {i: [] for i in range(num_users)}
    random_state = np.random.RandomState(seed)

    class_idx = 0
    for i in range(num_users):
        print("clinet = {} classes = ".format(i),end = "")
        for j in range(alpha):  # 每个client alpha 个类
            # print("classes = ",)
            print(class_idx,end = " ")
            train_selected = random_state.choice(train_idxs_classes[class_idx], int(num_client_train_sample / alpha), replace=False)
            train_dict_users[i] += train_selected.tolist()
            test_selected = random_state.choice(test_idxs_classes[class_idx], int(num_client_test_sample / alpha), replace=False)
            test_dict_users[i] += test_selected.tolist()
            train_idxs_classes[class_idx] = np.array(list(set(train_idxs_classes[class_idx]) - set(train_selected)))
            test_idxs_classes[class_idx] = np.array(list(set(test_idxs_classes[class_idx]) - set(test_selected)))
            class_idx += 1
            if class_idx == num_class: class_idx = 0
        print("")
        train_dict_users[i] = np.array(train_dict_users[i])
        test_dict_users[i] = np.array(test_dict_users[i])
    return train_dict_users, test_dict_users



def mnist_iid(train_dataset, test_dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    train_num_items = int(len(train_dataset)/num_users)
    train_dict_users, train_all_idxs = {}, [i for i in range(len(train_dataset))]
    test_num_items = int(len(test_dataset) / num_users)
    test_dict_users, test_all_idxs = {}, [i for i in range(len(test_dataset))]
    for i in range(num_users):
        train_dict_users[i] = set(np.random.choice(train_all_idxs, train_num_items,
                                                   replace=False))
        test_dict_users[i] = set(np.random.choice(test_all_idxs, test_num_items,
                                                   replace=False))
        train_all_idxs = list(set(train_all_idxs) - train_dict_users[i])
        test_all_idxs = list(set(test_all_idxs) - test_dict_users[i])
    return train_dict_users, test_dict_users



def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]   # 将数据按类别排序，从前到后是0-9类
    idxs = idxs_labels[0, :]    # 取出排序后的数据编号

    # divide and assign 2 shards/client
    for i in range(num_users):  # 为每个client分配 2 shards 数据
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



# 五个client的极端情况，每个client具有的类别的client都不具有
def cifar_extreme_case(train_dataset, test_dataset, num_users, alpha):
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    train_idxs_classes = [[] for _ in range(10)]
    test_idxs_classes = [[] for _ in range(10)]

    # 按照label排序，将同属一类的数据的位置索引放在一个列表中
    for i in range(50000):
        train_idxs_classes[train_labels[i]].append(i)
    for i in range(10000):
        test_idxs_classes[test_labels[i]].append(i)
    train_dict_users = {i: [] for i in range(num_users)}
    test_dict_users = {i: [] for i in range(num_users)}
    for i in range(5):
        class_idx = [2*i, 2*i+1]
        for j in class_idx:
            train_dict_users[i] += np.random.choice(train_idxs_classes[j], 300, replace=False).tolist()
            test_dict_users[i] += np.random.choice(test_idxs_classes[j], int(100), replace=False).tolist()
        train_dict_users[i] = np.array(train_dict_users[i])
        test_dict_users[i] = np.array(test_dict_users[i])
    return train_dict_users, test_dict_users



def mnist_imbalance(dataset,num_users,alpha):

    labels = dataset.train_labels.numpy()
    idxs_classes=[[] for _ in range(10)]

    # 按照label排序，将同属一类的数据的位置索引放在一个列表中
    for i in range(60000):
        idxs_classes[labels[i]].append(i)

    dict_users = {i: [] for i in range(num_users)}

    # 制造类别不平衡，选取5个类数据比较多，5个类数据比较少
    for i in range(num_users):
        majority = random.sample([_ for _ in range(10)], 5)
        for j in majority:
            dict_users[i] += np.random.choice(idxs_classes[j], 200,replace=False).tolist()

        minority = []
        for _ in range(10):
            if _ not in majority:
                minority.append(_)
        minority_num = int(250 / alpha)
        for j in minority:
            dict_users[i] += np.random.choice(idxs_classes[j], minority_num,replace=False).tolist()

        dict_users[i] = np.array(dict_users[i])

    return dict_users



def cifar_imbalance(dataset, num_users, alpha):
    labels = np.array(dataset.targets)
    idxs_classes = [[] for _ in range(10)]

    # 按照label排序，将同属一类的数据的位置索引放在一个列表中
    for i in range(50000):
        idxs_classes[labels[i]].append(i)

    dict_users = {i: [] for i in range(num_users)}

    # 制造类别不平衡，选取5个类数据比较多，5个类数据比较少
    for i in range(num_users):
        majority = random.sample([_ for _ in range(10)], 5)
        for j in majority:
            dict_users[i] += np.random.choice(idxs_classes[j], 200, replace=False).tolist()

        minority = []
        for _ in range(10):
            if _ not in majority:
                minority.append(_)
        minority_num = int(250 / alpha)
        for j in minority:
            dict_users[i] += np.random.choice(idxs_classes[j], minority_num, replace=False).tolist()

        dict_users[i] = np.array(dict_users[i])

    return dict_users



def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

# iid数据构建方法：将数据集打乱，随机抽取600个(num_users=100)划分给一个client。
def cifar_iid(train_dataset, test_dataset, num_users):
    """
    Sample I.I.D. client data from cifar10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    train_num_items = int(len(train_dataset)/num_users)
    train_dict_users, train_all_idxs = {}, [i for i in range(len(train_dataset))]
    test_num_items = int(len(test_dataset) / num_users)
    test_dict_users, test_all_idxs = {}, [i for i in range(len(test_dataset))]
    for i in range(num_users):
        train_dict_users[i] = set(np.random.choice(train_all_idxs, train_num_items,
                                                   replace=False))
        test_dict_users[i] = set(np.random.choice(test_all_idxs, test_num_items,
                                                   replace=False))
        train_all_idxs = list(set(train_all_idxs) - train_dict_users[i])
        test_all_idxs = list(set(test_all_idxs) - test_dict_users[i])
    return train_dict_users, test_dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)   # 生成数据集的索引
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    '''
    np.vstack 将两个数组按垂直方向（行顺序）堆叠数组构成一个新的数组
    '''
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
