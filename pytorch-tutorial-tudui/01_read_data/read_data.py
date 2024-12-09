import torch
from torch.utils.data import Dataset
from PIL import Image  # Python Imaging Library
import os

from torch.utils.data.dataset import T_co


# Dataset：提供一种方式去获取数据及其label
# 包含了：如何获取每一个数据及其label，告诉我们总共有多少的数据
# DataLoader：为后面的网络提供不同的数据形式
#

# print(dir(Dataset))
# print(help(Dataset))

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir  # self指代了类的实例。从而在类中都可以访问这些属性
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path_list=os.listdir(self.path)

    """ x.__getitem__(y) <==> x[y] """
    def __getitem__(self, index):
        img_name=self.img_path_list[index]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path_list)

class MyDatasetSeparate(Dataset):
    def __init__(self,root_dir,image_dir,label_dir):
        self.root_dir=root_dir
        self.image_dir=image_dir
        self.label_dir=label_dir
        self.img_path=os.path.join(self.root_dir,self.image_dir)
        self.label_path=os.path.join(self.root_dir,self.label_dir)
        self.img_path_list=os.listdir(self.img_path)
        self.label_path_list=os.listdir(self.label_path)

    def __getitem__(self, index):
        img_name=self.img_path_list[index]
        img_item_path=os.path.join(self.root_dir,self.image_dir,img_name)
        img = Image.open(img_item_path)
        label_name=self.label_path_list[index]
        label_item_path=os.path.join(self.root_dir,self.label_dir,label_name)
        with open(label_item_path,mode='r') as f:
            label=f.readline()


        sample={'img':img,'label':label}
        return sample



    def __len__(self):
        assert len(self.img_path_list)==len(self.label_path_list)
        return len(self.img_path_list)






def test_MyDataset():
    root_dir = "../dataset/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyDataset(root_dir, ants_label_dir)
    bees_dataset = MyDataset(root_dir, bees_label_dir)
    ant_img, ant_label = ants_dataset[0]

    ant_img.show()
    print(ant_label)
    # 仿造的数据集和真实的数据集进行训练
    train_dataset = ants_dataset + bees_dataset

def test_MyDatasetSeparate():

