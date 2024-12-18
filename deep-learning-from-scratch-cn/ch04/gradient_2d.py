# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 对于x0^2+x1^2的函数，x1的大小不影响x0的偏导数
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h,y)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h,y)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 还原值

    return grad

# 给定一个函数，在这个函数下求各个自变量在对应值的梯度
def numerical_gradient(f, X):

    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        # idx是索引，x是元素(一维数组)
        for idx, x in enumerate(X):
            print(f"idx={idx},x={x}")
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

# (x0)^2 + (x1)^2
def function_2(x):
    if x.ndim == 1:  # 输入的是[x0,x1]
        return np.sum(x**2)
    else:  # 输入的是[[x0,x1],[x0,x1],...]
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    # The numpy.meshgrid function returns two 2-Dimensional arrays representing the X and Y coordinates of all the points.
    # X是一个矩阵，每一行是x的一个完整的复制，Y也是一个矩阵，每一列是y的一个完整的复制
    X, Y = np.meshgrid(x0, x1)

    # flatten()将矩阵变成一维数组
    X = X.flatten()
    Y = Y.flatten()

    # 像素中的每个点都计算梯度
    grad = numerical_gradient(function_2, np.array([X, Y]))  # np.array()将X,Y合并成二维数组（第一行是X，第二行是Y）

    plt.figure()
    """
    X, Y: 矢量起点的坐标。
    U, V: 矢量的 x 与 y 分量。
    angles: 矢量的角度表示方式，通常设置为 "xy"。
    color: 矢量的颜色。=
    """
    # grad[0]是x0的梯度，grad[1]是x1的梯度
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()