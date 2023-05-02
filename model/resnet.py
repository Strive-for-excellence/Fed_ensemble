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
    def __init__(self, block, num_blocks, num_classes=10,drop_rate = 0.0,in_planes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
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


def ResNet18(num_classes=10,in_planes = 3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,in_planes = in_planes)
def ResNet34(num_classes=10,in_planes = 3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,in_planes = in_planes)
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
class LeNet(nn.Module):
    def __init__(self,drop_rate = 0.0,in_planes = 3,num_classes = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
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
class LeNet_Mnist(nn.Module):
    def __init__(self,drop_rate = 0.0,in_planes = 3,num_classes = 10):
        super(LeNet_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
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
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

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
