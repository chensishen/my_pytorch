import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
# input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.sigmoid1 = torch.nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid1(x)

net = Net()
# output = net(input)
# print(output)

writer = SummaryWriter("./my_relu")
step =0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = net(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()



