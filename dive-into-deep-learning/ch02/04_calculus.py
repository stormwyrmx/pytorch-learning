import torch

"""
gradient
partial derivative
derivative calculus/differential 
integral calculus
"""

x = torch.arange(4.0,requires_grad=True)  # 把梯度存在x.grad中
print(x)
print(x.grad)

y=2*torch.dot(x,x)
print(y)  # 隐式的构造了计算图
y.backward()  # 计算 y 对于所有依赖它的张量的导数，并将结果存储在这些张量的 .grad 属性中
print(x.grad)

x.grad.zero_()  # 梯度清零，不然会累加
y=torch.sum(x)
y.backward()
print(x.grad)



