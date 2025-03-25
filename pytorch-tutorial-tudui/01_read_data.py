from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs")

"""
一般是Image.open()后，进行transform（一般会包含ToTensor）
"""

# Dataset：提供一种方式去获取数据及其label
# 包含了：如何获取每一个数据及其label，告诉我们总共有多少的数据
# DataLoader：为后面的网络提供不同的数据形式
#

# print(dir(Dataset))
# print(help(Dataset))

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    # 使用__getitem__方法可以通过索引访问数据集中的元素
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.readline()

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        # 返回一个字典，字典中包含了图片和标签(哪个脑残想到返回字典的)
        return sample

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "data/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    # writer.add_image('error', train_dataset[119]['img'])
    # writer.close()
    for i, data in enumerate(dataloader):
        # 调用的是MyData的__getitem__方法，这里返回了字典。所以imgs是第一个item的key，labels是第二个item的key
        imgs, labels = data
        print(type(data))
        # print(i, data['img'].shape)
        print(i,imgs,labels)
        #  img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
        writer.add_image("train_data_b2", data[imgs], i, dataformats='NCHW')
        # writer.add_image("train_data_b2", make_grid(data['img']), i)

    writer.close()



