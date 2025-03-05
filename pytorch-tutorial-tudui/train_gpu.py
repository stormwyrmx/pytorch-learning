# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
import model
from torch import nn
from torch.utils.data import DataLoader

"""
对网络模型、损失函数、数据调用cuda或者to(device)方法
"""

# 定义训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
wengNet = model.WengNet().to(device)
wengNet.load_state_dict(torch.load('./saves/wengNet_gpu.pth',weights_only=True))
# wengNet = torch.load('./saves/wengNet_gpu.pth').to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(wengNet.parameters(), lr=learning_rate)


def train(dataloader, wengNet, loss_function, optimizer):
    # 训练步骤开始，这些都是最基础的
    size=len(dataloader.dataset)
    wengNet.train()
    for batch, (imgs, targets) in enumerate(dataloader):
        imgs, targets = imgs.to(device), targets.to(device)

        # Compute prediction error
        outputs = wengNet(imgs)
        loss = loss_function(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = (batch+1) * len(imgs)
            print(f" Loss: {loss.item()},训练次数：[{current}/{size}],")


def test(dataloader, wengNet, loss_function):
    # 测试步骤开始
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    wengNet.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = wengNet(imgs)
            test_loss += loss_function(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).type(torch.float).sum()

    test_loss /= num_batches
    correct /= size
    print(f"整体测试集上的Loss: {test_loss}")
    print(f"整体测试集上的正确率: {correct*100}")


# 设置训练网络的一些参数
# 训练的轮数
epochs = 10
for i in range(epochs):
    print(f"-------第 {i+1} 轮训练开始-------")
    train(train_dataloader, wengNet, loss_function, optimizer)
    test(test_dataloader, wengNet, loss_function)
print("训练完成")


torch.save(wengNet.state_dict(), "./saves/wengNet_gpu.pth")
print("模型已保存")
