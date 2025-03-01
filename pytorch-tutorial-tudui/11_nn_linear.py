# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 这里一定要drop_last=True，因为最后面的满足不了196608
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

wengNet = WengNet()
# summary_writer = SummaryWriter(log_dir="logs")

for i,data in enumerate(dataloader):
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = wengNet(output)
    print(output.shape)
    # summary_writer.add_images("input", imgs, i)
    # AssertionError: size of input tensor and input format are different. tensor shape: (10,), input_format: NCHW
    # summary_writer.add_images("output_linear", output, i)