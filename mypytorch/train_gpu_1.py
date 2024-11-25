# 网路网络模型
# 数据（输入，标注）
# 损失函数
# .cuda
import time
from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 打印长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))

# 利用dataloader，加载数据
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True,drop_last=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
net = Net()
if torch.cuda.is_available():
    net = net.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 100

#增加tensorboard
writer = SummaryWriter('./my_train_model')

start_time = time.time()
for i in range(epoch):
    print("--------------第{}轮训练开始---------------".format(i + 1))

    #训练步骤开始
    net.train()
    for data in train_dataloader:
        imgs, labels = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("结束时间：{}".format(end_time - start_time))
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs = net(imgs)
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_accuracy += accuracy
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()

    print("整体测试集上的loss:{}".format(total_test_loss))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)

    print("整体测试级上的准确率:{}".format(total_accuracy/train_data_size))
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)

    torch.save(net, 'my_train_model{}.pth'.format(total_test_step))
    print("模型{}被保存".format(total_test_step))
    total_test_step += 1

writer.close()