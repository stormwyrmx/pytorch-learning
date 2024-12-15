import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


# # 创建一个工具类
# tool = transforms.ToTensor()
# # 利用这个类将图片转换为张量
# tensor = tool(Image.open("dataset/train/ants/5650366_e22b7e1065.jpg"))
# print(tensor)
# # print(torch.tensor(Image.open("dataset/train/ants/5650366_e22b7e1065.jpg")))


class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
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

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.readline()

        if self.transform:
            img = transform(img)

        # 会返回一个元组，元组中包含了图片和标签
        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
root_dir = "data/train"
image_ants = "ants_image"
label_ants = "ants_label"
ants_dataset = MyData(root_dir, image_ants, label_ants, transform=transform)
image_bees = "bees_image"
label_bees = "bees_label"
bees_dataset = MyData(root_dir, image_bees, label_bees, transform=transform)





