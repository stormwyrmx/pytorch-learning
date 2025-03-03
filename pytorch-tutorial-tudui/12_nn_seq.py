# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Sequential：最适合网络结构是简单顺序、无分支的情况（如经典 MLP、简单 CNN、resnet block 等）。
        ModuleList：若网络中存在分支、跳跃连接，或者需要在 forward 不同路径调用不同层，则通过 ModuleList 存储子模块，然后在 forward 中自由编写调用逻辑。
        """
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            # 在默认条件，即：start_dim=1,end_dim=-1时，保留批次维度(第一维)，将剩余维度(通道、高、宽)展平成一维向量
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

wengNet = WengNet()
print(wengNet)
input = torch.ones((64, 3, 32, 32))
output = wengNet(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(wengNet, input)
writer.close()
