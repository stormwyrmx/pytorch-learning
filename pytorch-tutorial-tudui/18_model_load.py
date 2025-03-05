# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import _17_model_save
import torchvision

# 方式1，加载模型
# 当使用weights_only=True时，PyTorch只允许加载特定的安全数据类型，而完整的模型类定义（如torchvision.models.vgg.VGG）不在默认允许的类型列表中。
model = torch.load("./saves/vgg16_method1.pth",weights_only=False)
print("model1=",model)

# 方式2，加载模型
model2 = torchvision.models.vgg16(weights=None)
# 定义模型结构后，加载模型参数（状态字典）
model2.load_state_dict(torch.load("./saves/vgg16_method2.pth",weights_only=True))
print("model2=",model2)
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1,要让程序访问到定义好的模型（这里采用了import的方式）
model3 = torch.load('./saves/weng_method1.pth',weights_only=False)
print("model3=",model3)