# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import model
import torch

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 打印数据集的大小,不能调用train_data.shape和train_dataloader.shape，因为没有这个方法
print(f"训练数据集的长度为：{len(train_data)}")
print(f"测试数据集的长度为：{len(test_data)}")

# 利用 DataLoader 来加载数据集，在默认情况下使用顺序采样方式
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
wengNet = model.WengNet()

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2  # 0.01
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

for i in range(epoch):
    print(f"-------第 {i+1} 轮训练开始-------")

    # 训练步骤开始
    # This has any effect only on certain modules.
    # e.g.ik:class:`Dropout`, :class:`BatchNorm`,
    wengNet.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = wengNet(imgs)
        # 没必要再在模型里写loss（这样不是嵌了两层），直接nn调用就行了。
        loss = loss_function(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        # 通常来说，训练过程中每个mini - batch（迭代）都会计算loss，用于梯度反向传播和参数更新
        # 这里因为是为了演示，所以每100次迭代打印一次loss，pytorch官方也是每100次batch打印一次loss
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试步骤开始
    # 在整个epoch结束后，计算accuracy
    # 测试训练集、测试集、验证集，这里只测试了测试集
    wengNet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = wengNet(imgs)
            loss = loss_function(outputs, targets)
            # total_test_loss = total_test_loss + loss.item()
            # todo 因为这里每次只能求一个batch_size的正确率，所以要累加。能不能直接不需要data in dataloader。直接整个推理一遍
            # 这里就是每一个batch_size都要求loss和accuracy，然后加起来平均
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    # print(f"测试次数：{total_test_step}, Loss: {total_test_loss/total_test_step}")
    # writer.add_scalar("test_loss", total_test_loss/total_test_step, total_test_step)

    # 分类问题中，正确率 = 正确预测的数量 / 测试集的大小
    print(f"整体测试集上的正确率: {total_accuracy/len(test_data)}")
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test_step)
    total_test_step = total_test_step + 1


# 保存模型
torch.save(wengNet, f"./saves/wengNet_cpu.pth")
print("模型已保存")
writer.close()
