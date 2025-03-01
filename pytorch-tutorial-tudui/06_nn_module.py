# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn

class WengNet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x + 1
        return y

wengNet = WengNet()
x = torch.tensor(1.0)
# 因为集成的nn.Module中forword方法是__call__()方法的实现，可调用对象会调用__call__()方法
# 在 __call__ 方法内部，nn.Module 会自动调用 forward 方法
print(wengNet.forward(x))
output = wengNet(x)
print(output)

