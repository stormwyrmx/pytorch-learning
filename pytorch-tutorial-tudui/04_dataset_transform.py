import torchvision
from numpy.ma.core import shape
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Optional

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
# print(test_set[0])
# print(test_set.classes)

img, target = test_set[0]
print(img.shape)
print(target)
print(test_set.classes[target])
# ToPILImage 是一个可调用的转换类，其构造函数（init）用于设置参数，而图像转换操作发生在它的 call 方法中
# 这表示先实例化一个 ToPILImage 对象，然后用括号调用该对象，将 img 作为参数传入 call 方法进行转换，最后调用 show() 显示生成的 PIL 图像。
torchvision.transforms.ToPILImage()(img).show()

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()