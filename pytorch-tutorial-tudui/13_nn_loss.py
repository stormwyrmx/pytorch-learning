# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torch.nn as nn
from numpy import dtype

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss(reduction='sum') # default is 'mean'
result = loss(inputs, targets)
print(result)

loss_mse = nn.MSELoss(reduction='mean')
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.6, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
# 该损失函数内部结合了 nn.LogSoftmax 和 nn.NLLLoss。适用于多分类问题
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)