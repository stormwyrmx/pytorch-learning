# coding: utf-8
import os
import sys
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
# sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

# 使用梯度下降法计算损失函数相对于权重的梯度
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()
# w是一个参数，net.loss(x,t)是返回值。其中，net.loss中用的是self.init中的W
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
