import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        x = self.model1(x)
        return x
loss = torch.nn.CrossEntropyLoss()

net = Net()
print(net)
for data in dataloader:
    imgs,targets = data
    outputs = net(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print("ok")

