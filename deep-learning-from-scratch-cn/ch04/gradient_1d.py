# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4  # 科学计数法表示浮点数，表示0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    slope = numerical_diff(f, x)
    print(slope)  # 输出导数的值，即在x=5上的切线的斜率。下面要画出这条切线
    """
    计算直线的斜率 slope
    选择直线上任意一点 (x, y)
    使用公式 intercept = y - slope*x 计算截距。
    """
    intercept = f(x) - slope*x  # 计算切线在x=5上的截距
    return lambda t: slope*t + intercept
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
