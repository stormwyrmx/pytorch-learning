# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
import torch.nn as nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class WengNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 池化可以通过保留重要特征来减小数据量，加快训练速度。一般卷积后面会加一个池化
        # 池化不改变channel数
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

wengNet = WengNet()

writer = SummaryWriter("logs")

for i,data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images("input", imgs, i)
    output = wengNet(imgs)
    writer.add_images("output_maxpool", output, i)

writer.close()