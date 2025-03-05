# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn

# 搭建神经网络
class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model1(x)


if __name__ == '__main__':
    # 测试网络的正确性
    wengNet = WengNet()
    input = torch.ones(64, 3, 32, 32)
    output = wengNet(input)
    print(output.shape)