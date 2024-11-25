import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from my_train_model import *

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

# 创建网络模型
net = Net()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

#增加tensorboard
writer = SummaryWriter('./my_train_model')


for i in range(epoch):
    print("--------------第{}轮训练开始---------------".format(i + 1))

    #训练步骤开始
    net.train()
    for data in train_dataloader:
        imgs, labels = data
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            outputs = net(imgs)
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()

    print("整体测试集上的loss:{}".format(total_test_loss))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    print("整体测试级上的准确率:{}".format(total_accuracy/train_data_size))
    writer.add_scalar('test_accuracy', total_accuracy/train_data_size, total_test_step)
    torch.save(net, 'my_train_model{}.pth'.format(total_test_step))
    print("模型{}被保存".format(total_test_step))
    total_test_step += 1

writer.close()