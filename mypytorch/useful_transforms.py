import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# img_path = r"train/ants_image/0013035.jpg"
# cv_img = cv2.imread(img_path)

# ToTensor
trans_totensor = transforms.ToTensor()
# img_tensor = trans_totensor(cv_img)
# writer.add_image("ToTensor", img_tensor)

# Normalize
# print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize([1, 3, 5], [9, 7, 5])
# img_norm = trans_norm(img_tensor)
# writer.add_image("Normalize", img_norm,2)
# print(img_tensor[0][0][0])

# compose
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),       # 调整图片大小为 256x256
#     transforms.CenterCrop(224),         # 中心裁剪为 224x224
#     transforms.ToTensor(),              # 转换为张量，并归一化到 [0, 1]
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
# ])
#
from PIL import Image
img = Image.open(r"train/ants_image/0013035.jpg")
# transformed_img = transform(img)
# writer.add_image("Norm", transformed_img,1)

# Resize
# print(img.size)
# trans_resize = transforms.Resize((512,512))
# img_resized = trans_resize(img)
# print(img_resized.size)
img_tensor = trans_totensor(img)
writer.add_image("Resize", img_tensor)

# Compose - resize - 2
# 指定单个整数,按照比例调整短边到该值，同时长边根据比例自动缩放。
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_resize_2(img_tensor)
img_compose = trans_resize_2(img_resize_2)
writer.add_image("Resize", img_tensor,1)


# RandomCrop
trans_random = transforms.RandomCrop((100,50))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_compose_2 = trans_compose_2(img)
    writer.add_image("RandomCrop_2", img_compose_2,i)

writer.close()













