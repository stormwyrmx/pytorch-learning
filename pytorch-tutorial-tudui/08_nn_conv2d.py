# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
# from torch import nn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class WengNet(nn.Module):
    def __init__(self):
        super().__init__()                   # 卷积核是学出来的
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # 通过Python的__call__机制间接调用forward方法，会激活完整的PyTorch模块调用流程
        # 几乎所有情况下都应该使用self.conv1(x)，直接调用forward可能导致一些PyTorch功能不可用
        x = self.conv1(x)
        return x

wengNet = WengNet()
writer = SummaryWriter("logs")

for i,data in enumerate(dataloader):
    imgs, targets = data
    """
    wengNet(imgs) -> nn.Module.__call__() -> WengNet.forward(imgs)
    进入PyTorch的内部调用机制。检查模型的训练/评估状态。激活已注册的钩子函数(hooks)。准备计算图追踪(用于自动求导)
    """
    output = wengNet(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, i)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]
    # 6个channel翻倍是因为图片是三个通道的RGB
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output_conv", output, i)


writer.close()


