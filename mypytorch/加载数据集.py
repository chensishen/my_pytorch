from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os


# dar_path = r'G:\BaiduNetdiskDownload\数据集\hymenoptera_data\hymenoptera_data\train\ants'
#
#
# class MyData(Dataset):
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(root_dir, label_dir)
#         self.img_path = os.listdir(self.path)
#
#     def __getitem__(self, idx):
#         imf_name = self.img_path[idx]
#         img_item_path = os.path.join(self.path, imf_name)
#         img = Image.open(img_item_path)
#         label = self.label_dir
#         return img, label
#
#     def __len__(self):
#         return len(self.img_path)
#
#
# root_dir = r'G:\BaiduNetdiskDownload\数据集\hymenoptera_data\hymenoptera_data\train'
# ants_label_dir = r'ants'
# bees_label_dir = r'bees'
#
# ants_dataset = MyData(root_dir, ants_label_dir)
# bees_dataset = MyData(root_dir, bees_label_dir)
#
# train_dataset = ants_dataset + bees_dataset
#
# img,label = ants_dataset[1]
#
# img,label = bees_dataset[1]
# mg.show()
#
# print(len(train_dataset))
# print(len(bees_dataset))
# print(len(ants_dataset))
#
# img, label = train_dataset[123]
# img.show()
# img, label = train_dataset[124]
# img.show()

class MyData2(Dataset):
    def __init__(self, img_path, label_path):
        self.img_dir = img_path  # 保存图片目录
        self.label_dir = label_path  # 保存标签目录
        self.img_path = os.listdir(img_path)  # 图片文件名列表
        self.label_path = os.listdir(label_path)  # 标签文件名列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_item_path)
        label_file_name = img_name.split('.jpg')[0] + '.txt'  # 假设标签是文本文件
        label_file_path = os.path.join(self.label_dir, label_file_name)
        with open(label_file_path, "r", encoding="utf-8") as file:  # "utf-8" 适用于大多数文本文件
             label= file.read()
        return img, label

    def __len__(self):
        return len(self.img_path)


ants_dataset = MyData2(r'G:\BaiduNetdiskDownload\数据集\练手数据集\train\ants_image',
                      r'G:\BaiduNetdiskDownload\数据集\练手数据集\train\ants_label')

img,label = ants_dataset[0]
img.show()
print(label)

bees_dataset = MyData2(r'G:\BaiduNetdiskDownload\数据集\练手数据集\train\bees_image',
                      r'G:\BaiduNetdiskDownload\数据集\练手数据集\train\bees_label')
img,label = bees_dataset[0]
img.show()
print(label)


