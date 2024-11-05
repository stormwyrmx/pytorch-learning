import torch

# print(torch.cuda.is_available())
x=torch.arange(24)  # 一维张量，称之为向量
print(type(x))
print(x.shape)  # 一维，长度为22
print(x.numel())  # number of elements
print(x.reshape(3,2,4))  # reshape成3个2*4的矩阵（2*4的矩阵是2个4个元素的向量）


print(torch.zeros(2, 3, 4))
print(torch.zeros((2, 3, 4)))
print(torch.ones(2, 3, 4))
print(torch.tensor([[1, 2], [3, 4]]))









