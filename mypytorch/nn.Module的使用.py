import torch
import torch.nn as nn
import torch.nn.functional as F


# forward 前向传播

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        output = input +1
        return output

net = Net()
x = torch.tensor(1.0)
output = net(x)
print(output)




























