# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
wengNet = WengNet()
for data in dataloader:
    imgs, targets = data
    # 值得注意的是，虽然outputs本身不是概率分布（会有负的），但在使用nn.CrossEntropyLoss()时，损失函数内部会先对outputs应用log_softmax函数，
    # 将其转换为概率分布（每个样本的10个类别概率总和为1），然后再计算交叉熵损失
    # 模型输出的是没有用softmax函数归一化的概率，为了防止softmax函数的分子和分母在计算指数时数据溢出，将softmax函数整合到交叉熵损失函数里面了
    outputs = wengNet(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
