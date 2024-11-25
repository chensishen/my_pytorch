import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()

writer = SummaryWriter('./my_conv2d')

step = 0
for data in dataloader:
    images, labels = data
    output = net(images)
    print(images.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", images, step)
    # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3,30,30))
    writer.add_images("output", output, step)
    step += 1

writer.close()