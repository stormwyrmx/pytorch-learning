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
img.show()


writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()