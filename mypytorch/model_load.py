import torch
import torchvision
import model_save

# 方式一-》保存方式1，加载模型

# model = torch.load("vgg16_method1.pth", weights_only=False)
# print(model)

# 方式二-》保存方式1，加载模型
# vgg16 = torchvision.models.vgg16(weights = None)
# model = torch.load("vgg16_method2.pth", weights_only=False)
# vgg16.load_state_dict(model)
# print(model)
# print(vgg16)

# 陷阱1
model = torch.load("net_method1.pth", weights_only=False)
print(model)