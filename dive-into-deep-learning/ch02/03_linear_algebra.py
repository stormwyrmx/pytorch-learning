import torch
import numpy as np


"""
矩阵左边的每一行都代表一个产品的价目表，右边的每一列都代表一种做饭的方式，
那么所有可能的组合所最终产生的花费
"""
print("------------向量-------------")
x = torch.tensor([3.0,4.0])
y = torch.tensor([2])
print(x + y)
print(x * y)

print("------------矩阵-------------")
A = torch.arange(12,dtype=torch.float32).reshape(4,3)
print(A)
print(A[3,1])
print(len(A))
"""
向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。 
然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。
"""
print(A.shape)
print(A.T)

print("------------张量-------------")
B=A.clone()
print(A+B)
print(A*B)  # Hadamard product
print(A+2)
print(A*2)

print("------------维度-------------")
A=torch.arange(40,dtype=torch.float32).reshape(2,5,4)
print(A.sum())
print(A.sum(dim=0))  # 按照哪个轴求和就会丢掉该维度。结果维度就是5*4
print(A.sum(dim=1))  # 结果维度就是2*4
print(A.sum(dim=[1,2]))  # 结果维度就是2
print(A.mean(dim=1))  # =print(A.sum(dim=1)/A.shape[1])
# torch.unsqueeze(A,0)

A_sum=A.sum(dim=0,keepdim=True)  # 二阶不保留一样可以，因为一维向量是1*n阶矩阵
print(A_sum)  # 保留维度，会把那一个维度变为1， 结果维度是2*1*4
print(A/A_sum)
print(A.cumsum(dim=1))  # 沿着行方向累加

print("------------点积、矩阵向量积、矩阵乘法-------------")
x,y=torch.ones(4),torch.ones(4)
print(x.dot(y))
print(torch.sum(x*y))
print(torch.mv(A[0],x))  # matrix-vector product
print(torch.mm(A[0],A[0].T))  # matrix-matrix product

print("------------范数-------------")
u=torch.tensor([3.0,4])
print(torch.norm(u))  # L2范数
print(torch.abs(u).sum())  # L1范数
print(torch.norm(torch.ones(4,9)))  # Frobenius范数，相当于拉成一维向量后的L2范数

print("------------练习-------------")
print(len(A))
print(torch.norm(torch.ones(2,4,5)))






