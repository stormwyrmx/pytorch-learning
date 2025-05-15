# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from _17_model_save import WengNet
import torchvision

# 方式1，加载模型
# 当使用weights_only=True时，PyTorch只允许加载特定的安全数据类型，而完整的模型类定义（如torchvision.models.vgg.VGG）不在默认允许的类型列表中。
# weights_only=True 参数的含义是告诉 PyTorch 在加载模型时只加载模型的权重（参数），而不加载整个模型结构、类定义等其他信息。
model = torch.load("./saves/vgg16_method1.pth",weights_only=False)
print("model1=",model)

# 方式2，加载模型
model2 = torchvision.models.vgg16(weights=None)

# 定义模型结构后，加载模型参数（状态字典）。定义模型后，再用文件里的东西overwrite掉我们自己的东西
# Copy parameters and buffers from :attr:`state_dict` into this module and its descendants
model2.load_state_dict(torch.load("./saves/vgg16_method2.pth",weights_only=True))
print("model2=",model2)
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1,要让程序访问到定义好的模型（这里采用了import的方式）
model3 = torch.load('./saves/weng_method1.pth',weights_only=False)
print("model3=",model3)