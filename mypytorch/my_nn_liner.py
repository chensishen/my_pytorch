import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(196608,10)
    def forward(self, x):
        return self.linear1(x)

net = Net()

# for data in dataloader:
#     images, labels = data
#     print(images.shape)
#     outputs = torch.reshape(images,(1,1,1,-1))
#     print(outputs.shape)

for data in dataloader:
    imags,targets = data
    print(imags.shape)
    output = torch.flatten(imags)
    print(output.shape)
    output = net(output)
    print(output.shape)










