# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# pylab` is a historic interface and its use is strongly discouraged.
# The equivalent replacement is `matplotlib.pyplot


def step_function(x):
    return np.array(x > 0, dtype=np.int32)

X = np.arange(-5.0, 5.0, 0.1)
print(X>0)
Y = step_function(X)
print(Y)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
