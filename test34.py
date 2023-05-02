# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from torchstat import stat
# Model
print('==> Building model..')
model = torchvision.models.resnet18(pretrained=False)

dummy_input = torch.randn(3, 224, 224)
stat(model,input_size=(3,224,224))
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
