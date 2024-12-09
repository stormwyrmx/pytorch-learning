from numpy.ma.core import shape
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2

writer = SummaryWriter("logs")

image_path1 = "data/train/ants_image/6240329_72c01e663e.jpg"
image_path2 = "data/train/bees_image/16838648_415acd9e3f.jpg"

img_PIL = Image.open(image_path1)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
img_cv2 = cv2.imread(image_path2)
print(type(img_cv2))
print(shape(img_cv2))

# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
writer.add_image("train", img_array, 1, dataformats='HWC')
writer.add_image("train", img_cv2, 2, dataformats='HWC')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()