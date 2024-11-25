import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 32, 5,padding=2)
        # self.maxpool1 = torch.nn.MaxPool2d(2)
        # self.conv2 = torch.nn.Conv2d(32, 32, 5,padding=2)
        # self.maxpool2 = torch.nn.MaxPool2d(2)
        # self.conv3 = torch.nn.Conv2d(32, 64, 5,padding=2)
        # self.maxpool3 = torch.nn.MaxPool2d(2)
        # self.flatten = torch.nn.Flatten()
        # self.fc1 = torch.nn.Linear(1024, 64)
        # self.fc2 = torch.nn.Linear(64, 10)

        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.model1(x)
        return x

net = Net()
print(net)
input = torch.ones((64,3,32,32))
output = net(input)
print(output.shape)

writer = SummaryWriter('./my_seq')
writer.add_graph(net, input)
writer.close()


















