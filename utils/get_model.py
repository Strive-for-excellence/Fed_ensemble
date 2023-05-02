from model import *

def Get_model(args):
    model = ""
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        model =  LeNet_Mnist(num_classes=args.num_classes,in_planes=1)
    elif args.dataset == 'cifar10':
        model =  LeNet(num_classes=args.num_classes,in_planes=3)
    elif args.dataset == 'cifar100':
        model =  ResNet8(num_classes=args.num_classes)
    else:
        print('error model input')
        exit(0)
    return model