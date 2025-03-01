# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
import torch.nn as nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 激活函数是加入非线性因素，线性并不能很好的拟合现实的情况，加入非线性因素可以增强拟合能力
        self.relu1 = ReLU(inplace=True)
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        self.relu1(input)
        # output = self.sigmoid1(input)
        # return output

wengNet = WengNet()

writer = SummaryWriter("./logs")

for i,data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images("input", imgs, global_step=i)
    wengNet(imgs)
    writer.add_images("output_relu", imgs, i)

writer.close()


