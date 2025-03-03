# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

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


writer = SummaryWriter("logs")
loss = nn.CrossEntropyLoss()
wengNet = WengNet()
optim = torch.optim.SGD(wengNet.parameters(), lr=0.01)  # lr最好一开始大，然后慢慢减小
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        # pyTorch 会根据网络结构（卷积、池化、全连接等运算）自动构建一张计算图
        outputs = wengNet(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        # 以逆拓扑顺序（自后往前）遍历计算图，计算**损失**对每个中间节点（操作）的偏导数，并将结果逐层反传回去，最终得到损失对每个可训练参数的梯度。
        result_loss.backward()
        # 优化器根据梯度进行的优化，梯度是通过损失函数得来的。让损失函数最小
        optim.step()
        running_loss = running_loss + result_loss

        writer.add_graph(wengNet, imgs)
    print(running_loss)

