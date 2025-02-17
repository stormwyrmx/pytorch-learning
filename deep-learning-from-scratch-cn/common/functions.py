# coding: utf-8
import numpy as np

count=0

def identity_function(x):
    return x


# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策，减去x数组中最大量 np.max()≠np.maximum()
    # np.maximum() performs element-wise comparison between two arrays and
    # returns a new array containing the maximum values from each pair of elements.
    return np.exp(x) / np.sum(np.exp(x))

# the loss function measures the difference between the predicted output of a model and the actual target values
# The goal of training a machine learning model is to minimize this loss function
# 均方误差(Mean Squared Error) 预测值与真实值之间差异的平方的平均值 1/n * Σ(y - t)^2
# 下列函数只适用于t是one-hot-vector，且只有1个数据的情况
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
"""
在实际应用中，尤其是训练神经网络时，这个平均因子n有时被省略或在其他地方处理（例如在批量处理时已经通过批量大小进行了归一化）。
因此，定义损失函数为 0.5 倍的平方误差和是一种常见的做法，主要是为了简化后续的梯度计算和优化过程。"""

# 交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # sum中的参数是一个数组，表示对应元素相加
    # global count
    # count+=1
    # if count%5000==0:
    #     print(count)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
