"""
.. Deep Residual Learning for Image Recognition:
    https://arxiv.org/abs/1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,drop_rate = 0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.drop_layer = nn.Dropout(p=self.drop_rate)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.drop_layer(out)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
def ResNet18_drop(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, drop_rate=0.5)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

class DNN(nn.Module):
    def __init__(self, in_dim,out_dim,n_hid,args):
        super(DNN, self).__init__()
        self.drop_rate = args.drop_rate
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, out_dim)
        self.drop_layer = nn.Dropout(p=self.drop_rate)
    def forward(self, x):
        # mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.in_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        # -----------------
        x = F.relu(x)
        # -----------------
        x = self.fc2(x)
        x = self.drop_layer(x)
        # -----------------
        x = F.relu(x)
        # -----------------
        out = self.fc3(x)

        return out
class LeNet_cifar10(nn.Module):
    def __init__(self,drop_rate = 0.0):
        super(LeNet_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop_layer = nn.Dropout(p=drop_rate)


    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = self.fc3(x)
        return x


# class LeNet_cifar10_for_ensemble(nn.Module):
#     def __init__(self, drop_rate=0.0):
#         super(LeNet_cifar10_for_ensemble, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
#         self.conv21 = nn.Conv2d(6, 16, kernel_size=5)
#         self.conv22 = nn.Conv2d(6, 16, kernel_size=5)
#         self.conv23 = nn.Conv2d(6, 16, kernel_size=5)
#         self.conv24 = nn.Conv2d(6, 16, kernel_size=5)
#
#         self.fc1 = nn.Linear(16 * 5 * 5 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#         self.drop_layer = nn.Dropout(p=drop_rate)
#
#     # 32*32 32-5+1 28
#     # 28*28 28/2 14
#     # 14*14 14-4 10
#     # 10*10 10/2 5
#
#     def forward(self, x):
#         x = func.relu(self.conv1(x))
#         x = func.max_pool2d(x, 2)
#
#         x1 = func.relu(self.conv21(x))
#         x1 = func.max_pool2d(x1, 2)
#         x1 = x1.view(x1.size(0),-1)
#
#         x2 = func.relu(self.conv22(x))
#         x2 = func.max_pool2d(x2, 2)
#         x2 = x2.view(x2.size(0),-1)
#
#         x3 = func.relu(self.conv23(x))
#         x3 = func.max_pool2d(x3, 2)
#         x3 = x3.view(x3.size(0),-1)
#
#         x4 = func.relu(self.conv24(x))
#         x4 = func.max_pool2d(x4, 2)
#         x4 = x4.view(x4.size(0),-1)
#         x = torch.cat((x1,x2,x3,x4),dim=1)
#         x = x.view(x.size(0), -1)
#         x = func.relu(self.fc1(x))
#         x = func.relu(self.fc2(x))
#         x = self.drop_layer(x)
#         x = self.fc3(x)
#         return x

class LeNet_cifar10_for_ensemble2(nn.Module):
    def __init__(self, drop_rate=0.0):
        super(LeNet_cifar10_for_ensemble2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)

        ls = []
        for i in range(4):
            ls.append(nn.Conv2d(6, 16, kernel_size=5))
        self.conv2 = nn.ParameterList(ls)


        self.fc1 = nn.Linear(16 * 5 * 5 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.drop_layer = nn.Dropout(p=drop_rate)

    # 32*32 32-5+1 28
    # 28*28 28/2 14
    # 14*14 14-4 10
    # 10*10 10/2 5

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        out = [conv(x) for conv in self.conv2]
        out = torch.cat(out, dim=1)
        x = out.view(out.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        # x = self.drop_layer(x)
        x = self.fc3(x)
        return x



class LeNet_mnist(nn.Module):
    def __init__(self):
        super(LeNet_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
        self.relu5 = nn.ReLU()
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.drop_layer(y)
        y = self.fc3(y)
        return y
if __name__ == '__main__':

    # net = ResNet18()
    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())
    net = DNN(256,10,1200)
    x = torch.randn(1,1,2,128)
    x = x.view(-1,256)
    y = net(x)
    print(y.size())
# test()
