# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from PIL import Image
from torch import nn
import model

"""
如果报错输入类型和权重类型不符的，因为你模型是在cuda上跑的，验证也要转cuda才行
"""
# 定义测试的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 加载数据集
image_path = "imgs/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

# 加载模型
wengNet = model.WengNet().to(device)
wengNet.load_state_dict(torch.load('./saves/wengNet_gpu.pth',weights_only=True))
# wengNet = torch.load("./saves/wengNet_gpu.pth")
print(wengNet)

# 模型预测
image = torch.reshape(image, (1, 3, 32, 32))
wengNet.eval()
with torch.no_grad():
    image = image.to(device)
    output = wengNet(image)

print(output.argmax(1))
