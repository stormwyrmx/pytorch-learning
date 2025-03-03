# coding: utf-8
import os
import sys
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 这里的load_mnist读入的数据集是在哪里的？
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
