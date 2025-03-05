# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import model
from torch import nn
from torch.utils.data import DataLoader
import time

"""
对网络模型、损失函数、数据调用cuda或者to(device)方法
"""

# 定义训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
# wengNet = model.WengNet()
wengNet = torch.load('./saves/wengNet_gpu.pth')
wengNet.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(wengNet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")

start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始，这些都是最基础的
    wengNet.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = wengNet(imgs)
        loss = loss_function(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    wengNet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = wengNet(imgs)
            # loss = loss_function(outputs, targets)
            # total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    # print("整体测试集上的Loss: {}".format(total_test_loss))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1


torch.save(wengNet, f"./saves/wengNet_gpu.pth")
print("模型已保存")
writer.close()
