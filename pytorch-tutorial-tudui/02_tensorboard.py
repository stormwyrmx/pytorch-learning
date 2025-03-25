import torch
from numpy.ma.core import shape
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2


"""
pytorch的tensor中的图像的数据格式是CHW，像素值范围是[0, 1]
opencv中的图像数据格式是HWC，像素值范围是[0, 255]
PIL读取的图片的通道（C）顺序是RGB，而opencv读取的图片的通道（C）顺序是BGR
"""

writer = SummaryWriter("logs")

image_path1 = "data/train/ants_image/6240329_72c01e663e.jpg"
image_path2 = "data/train/bees_image/16838648_415acd9e3f.jpg"

img_PIL = Image.open(image_path1)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
img_cv2=cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2RGB)
print(type(img_cv2))
print(img_cv2.shape)
# print(shape(img_cv2))

# ToTensor
#  Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
#  to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
transform_totensor = transforms.ToTensor()
tensor_image = transform_totensor(img_array)
print(tensor_image.shape,tensor_image.mean(),tensor_image.std())
tensor_cv2 = torch.tensor(img_cv2)
print(tensor_cv2.shape)

# Normalize
# 只能处理CHW格式的数据，所以需要先将数据转换为CHW格式。同时C一定是和mean中的元素个数相同
transforms_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
image_normalize = transforms_normalize(tensor_image)

# Resize
transforms_resize = transforms.Resize((256, 256))
image_resize = transforms_resize(img_PIL)
print(img_PIL.size)
print(image_resize.size)

# Compose
transforms_compose = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
image_compose = transforms_compose(img_PIL)

# RandomCrop
transforms_compose2 = transforms.Compose([transforms.RandomCrop(150), transforms.ToTensor()])
for i in range(10):
    image_compose2 = transforms_compose2(img_PIL)
    writer.add_image("randomCrop", image_compose2, i)

# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
writer.add_image("train", tensor_image, 1, dataformats='CHW')
writer.add_image("train", tensor_cv2, 2, dataformats='HWC')
writer.add_image("train", image_normalize, 3, dataformats='CHW')
writer.add_image("train", transforms.ToTensor()(image_resize), 4, dataformats='CHW')
writer.add_image("train", image_compose, 5, dataformats='CHW')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()