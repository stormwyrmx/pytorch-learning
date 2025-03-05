# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn

class WengNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(weights=None)
    print(vgg16)
    # 在PyTorch中，state_dict()是一个重要方法，它返回一个包含模型所有参数（权重和偏置）的Python字典。  具体来说：
    # 字典的键(keys)是参数的名称（如"features.0.weight"）
    # 字典的值(values)是对应的参数张量(tensor)
    print(vgg16.state_dict())

    # 保存方式1,模型结构+模型参数
    torch.save(vgg16, "./saves/vgg16_method1.pth")

    # 保存方式2，只保存模型参数（官方推荐，因为用的空间更小）
    torch.save(vgg16.state_dict(), "./saves/vgg16_method2.pth")

    # 陷阱1
    wengNet = WengNet()
    torch.save(wengNet, "./saves/weng_method1.pth")