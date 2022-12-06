# import time
# import tqdm
# import torch
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import  os
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# x = [5,6,7,8,9,10]
# a = [[0.377,0.396,0.435,0.474,0.482,0.523],
#      [0.296,0.289,0.449,0.505,0.550,0.584],
#      [0.376,0.481,0.512,0.536,0.587,0.597]]
# name = ['单独训练','FedMD', 'Ours']
#
# # for i in range(3):
# #     plt.plot(x,a[i],label = name[i])
# # plt.xlabel('epoch')
# # plt.ylabel('accuracy')
# # plt.legend()
# # plt.title('cifar10')
# # # plt.show()
# # plt.savefig('cifar10_path', dpi=480,bbox_inches='tight')
#
# x = [50,60,70,80,90,100]
# a = [[0.262,0.292,0.312,0.318,0.336,0.348],
#      [0.289,0.298,0.449,0.505,0.550,0.584],
#      [0.376,0.481,0.512,0.536,0.587,0.597]]
# for i in range(3):
#     plt.plot(x,a[i],label = name[i])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.title('cifar100')
# # plt.show()
# plt.savefig('cifar100_path', dpi=480,bbox_inches='tight')


def add(a,b):
    a = a+b
    return a
a = 1
b = 1
print(add(a,b))
print(a)

def max_value(node,alpha,beta):
    if is_leaf(node):
        return node.value()
    tmp = -inf
    for chl in node.child:
        tmp = max(tmp,min_value(chl,alpha,beta))
        if tmp >= beta:
            return tmp
        alpha = max(alpha,tmp)
    node.val = tmp
    return tmp
def min_value(node,alpha,beta):
    if is_leaf(node):
        return node.value()
    tmp = inf
    for chl in node.child:
        tmp = min(tmp,max(chl,alpha,beta))
        if tmp <= alpha:
            return tmp

        beta = min(beta,tmp)
    node.val = tmp
    return tmp

def max_value(node,alpha,beta):
    if is_leaf(node):
        return node.val
    tmp = -inf
    for child in node.child:
        tmp = max(tmp,min_value(child,alpha,beta))
        if tmp >= beta:
            return tmp
        alpha = max(tmp,alpha)
    return tmp
def min_value(node,alpha,beta):
    if is_leaf(node):
        return node.val
    tmp = inf
    for child in node.child:
        tmp = min(tmp,max_value(child,alpha,beta))
        if tmp <= alpha:
            return tmp
        beta = min(beta,tmp)
    return tmp