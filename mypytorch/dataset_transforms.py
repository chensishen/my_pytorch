import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((64, 64))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=True)

# print(test_set.classes)
# img,target = test_set[0]
# print(img,target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("p10")
for i in range(10):
    img, label = test_set[i]
    writer.add_image("test_img", img, i)
writer.close()