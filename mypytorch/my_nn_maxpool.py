import torch
import torch.nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


# 减小计算量

dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]])
#
# input = torch.reshape(input, (-1,1, 5, 5))
# print(input.shape)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=False)
    def forward(self, x):
        x = self.maxpool1(x)
        return x

net = Net()

writer = SummaryWriter("./my_maxpool")
step =0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = net(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()

