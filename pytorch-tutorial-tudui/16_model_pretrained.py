# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision
from torch import nn
import os

# train_data = torchvision.datasets.ImageNet("dataset", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

# 这里设置了windows的环境变量的TORCH_HOME为 "D:\ai\数据集"

vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16_false = torchvision.models.vgg16(weights=None)

print(vgg16_true)
print(vgg16_false)

train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 有feature、avgpool、classifier
# avgpool的输入为features的输出，确保输出固定为7×7大小
# classifier将特征映射到类别概率。输入为展平的特征向量(25088维)，输出为类别概率(1000维)
# 这里在修改网络结构
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 修改最后一层的out_features
print(vgg16_false)


