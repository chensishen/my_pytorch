import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# train_data = torchvision.datasets.ImageNet("./data_img_net",split='train',download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_False = torchvision.models.vgg16(weights=None)
vgg16_True = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
print('ok')

print(vgg16_True)

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

# vgg16_True.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16_True)

vgg16_True.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_True)

print(vgg16_False)
vgg16_False.classifier[6] = nn.Linear(4096, 10)
print(vgg16_False)
