import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((64, 64))
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform)

test_loader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集第一张图片以及target
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images(f"test_imgs/epoch_{epoch}", imgs, step)
        step += 1
writer.close()
