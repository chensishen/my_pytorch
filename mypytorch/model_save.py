import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1,模型结构+模型参数
# torch.save(vgg16, 'vgg16_method1.pth')

#保存方式2，模型参数（官方推荐）
# torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 陷阱
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
net = Net()
torch.save(net, 'net_method1.pth')









