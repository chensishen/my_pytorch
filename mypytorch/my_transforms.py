from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

# 图片经过工具后输出结果
# 创建具体的工具tool = transforms.ToTensor()
# 实用工具result = tool(input)


# python的用法-》tensor数据类型
# 通过transforms。ToTensor去看俩个问题
# 1.transforms该如何使用（python）
# 2.为什么我们想要Tensor数据类型
# 含有神经网络的一些参数

writer = SummaryWriter("logs")

# img_path = r"train/ants_image/0013035.jpg"
# cv_img = cv2.imread(img_path)

img_path = r"G:\BaiduNetdiskDownload\数据集\练手数据集\train\ants_image\0013035.jpg"
img = Image.open(img_path)

# 1.transforms该如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()


















