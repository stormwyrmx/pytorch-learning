import torchvision
# 准备的测试数据集
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,transform=torchvision.transforms.ToTensor(), download=True)
print(type(test_set),type(test_set[0]))
# 这里的shuffle改变的是每个epoch的数据顺序，而不是batch之间的数据顺序
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
print(type(test_loader),type(test_loader.dataset[0]))
# 测试数据集中第一张图片及target，test_set[0]返回的是一个元组，包含了图片和标签
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    for i,data in enumerate(test_loader):
        # 调用 __iter__() 方法：
        # 会为每个批次数据附加一个从0开始的索引
        # 循环中，i获取当前批次的索引号，data获取当前批次的数据
        # 返回一个迭代器对象（DataLoaderIter 或其自定义实现）。
        # 这个迭代器会基于 dataset、batch_size 和 shuffle 等参数动态生成批次。
        # 底层调用dataset的__getitem__方法，这里返回的是一个元组，元组中包含了图片和标签
        imgs, targets = data  # 从data元组中取出数据
        print(imgs.shape)
        print(targets)
        # 注意这里是add_images（默认的dataformats是NCHW），而不是add_image
        writer.add_images("Epoch: {0}".format(epoch), imgs, i)

writer.close()
