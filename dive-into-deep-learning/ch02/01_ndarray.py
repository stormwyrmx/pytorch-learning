import torch

print(torch.cuda.is_available())
# dir()就是查看当前模块的所有属性和方法，help()是查看函数或者模块的详细说明
print("================tensor=================")
x = torch.arange(24)  # 一维张量，称之为向量
print(type(x))
print(x.shape)  # 等价于print(x.size()) 一维，长度为22
print(x.numel())  # number of elements
print(x.view(2, 3, 4))  # view是reshape的别名，reshape成2*3*4的矩阵
print(x.reshape(3, 2, 4))  # reshape成3个2*4的矩阵（2*4的矩阵是2个4个元素的向量）
print(torch.zeros(2, 3, 4))
print(torch.zeros((2, 3, 4)))
print(torch.ones(2, 3, 4))
print(torch.tensor([[[1, 2, 3, 4], [3, 4, 2, 1], [4, 5, 6, 7]]]).shape)  # Channel 1,Height 3,Width 4。1层3行4列的矩阵
print(torch.randn(3, 4))  # 生成一个3*4的矩阵，每个元素都是从标准正态分布中随机采样的

# 对于任意具有相同形状的张量， 常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算
print("================element-wise compute=================")
a=torch.tensor([[1.0,2,3],[4,5,6]])
b=torch.tensor([[1,5,6],[7,8,9]])
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a**b)
print(torch.exp(a))
print(torch.cat((a,b),dim=0))  # concatenate 可以理解为，dim 的维度是从最外层“[”开始算的 ，如果是三维，dim=2才是将列进行合并。
print(torch.cat((a,b),dim=1))  # 0 和 1 这种表示法与 .shape的结果是对应的，0代表沿着第0个维度，1代表沿着第1个维度
print(a==b)  # 按照逻辑运算符的规则，返回一个布尔张量
print(a.sum())  # 对张量中的所有元素进行求和，会产生一个单元素张量

print("================broadcasting=================")
c=torch.arange(2).reshape(2,1)
d=torch.arange(3).reshape(1,3)
print(c+d)

print("================index and slice=================")
# 拿出来的都是第1个元素，只不过是二维所以是行
print(a[1][2])  # 第1行第0列 It first selects the 1st row (a[1]), which is a tensor, and then selects the 2nd element of that row ([2]).
print(a[1,2])  # 第1行第2列 This accesses the element at the 1st row and 2nd column directly in one step using a single indexing operation.
print(a[0:2,0])  # 第0行到第2行（不包含第2行），第0列
print(a[:,1:3])  # 第1列到第3列（不包含第3列）
a[1,2]=9
print(a)
a[:,1:3]=555  # :表示沿轴0的所有元素
print(a)

print("================saving memory=================")
print(id(a))
c=a
print(id(c))

z=torch.zeros_like(a)
print(id(z))
z[:]=a+b
print(z)
print(id(z))

print(id(a))
a[:] = a + b  # 这里的[:]是指定a的所有元素，如果不加[:]，a的引用会指向新的对象，而不是原来的对象
print(id(a))
a+=b
print(id(a))

print("================conversion with numpy=================")
# tensor有数学上的概念，numpy只有计算机的概念
a_numpy = a.numpy()
print(type(a_numpy))
print(a_numpy)
a = torch.tensor(a_numpy)
print(type(a))
print(a)
f=torch.tensor([3.5])
print(f,f.item(),float(f),int(f))












