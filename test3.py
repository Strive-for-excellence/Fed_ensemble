import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as pyplot
from model.resnet import  *


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
train_loader = torch.utils.data.DataLoader(trainset,batch_size=256,shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,batch_size=256,shuffle=False)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
# print('device=',device)

model = LeNet_cifar10_for_ensemble2()
# print(model.state_dict().keys())
# print(model)
for name, param in model.named_parameters():
    print(name)
# print(model.named_parameters())
model.to(device)
# model = LeNet_mnist()
optimizer  = torch.optim.SGD(model.parameters(),lr=0.01)
criterion = torch.nn.CrossEntropyLoss().to(device)
model.train()

Epochs = 200
def inference(model,test):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels.long())
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
            total += len(labels)

        accuracy = correct / total
        loss = loss / total
    return accuracy,loss
def train():
    train_loss = []
    test_acc = []
    test_loss = []
    for iter in range(Epochs):
        print(f'local epoch: {iter}')
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels.long())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item() / len(labels))
        print('train_loss = ',sum(batch_loss) / len(batch_loss))
        train_loss.append(sum(batch_loss) / len(batch_loss))
            # print('train_acc = ',l)
        if iter % 10 == 0:
            acc,loss = inference(model,testset)
            test_acc.append(acc)
            test_loss.append(loss)
            print('epoch = {},acc = {},loss = {}'.format(iter,acc,loss))
    pyplot.plot(test_acc)
    plt.title('test acc')

# train()

