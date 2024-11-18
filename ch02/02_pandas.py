import pandas as pd
import os

import torch
from pandas import DataFrame

"""
pandas预处理原始数据，并将原始数据转换为张量格式
"""
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs:DataFrame = data.iloc[:, 0:2]
outputs:DataFrame = data.iloc[:, 2]

inputs.fillna(inputs.mean(numeric_only=True), inplace=True)
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True,dtype=int)
# one-hot encoding
#  It creates new binary columns (0s and 1s) for each category in the original variable.
#  Each category in the original column is represented as a separate column,
#  where a value of 1 indicates the presence of that category, and 0 indicates its absence.
# convert categorical variables into dummy/indicator variables.
# It does not convert numerical values into dummy variables
print(inputs)  # col的名字再加上了_后面的值，1表示有，0表示没有

# print(torch.tensor(inputs.to_numpy()))
x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x,y)  # python默认使用float64，深度学习一般用float32

print("------------练习------------")
# 删除缺失值最多的列
count_max=0
label=""
for key in data.keys():
    if count_max<data.loc[:,key].isna().sum():
        count_max=data.loc[:,key].isna().sum()
        label=key

print(data.drop(axis=1,columns=label))
print(data)

