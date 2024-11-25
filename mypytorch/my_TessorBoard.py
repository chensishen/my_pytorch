
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = r"G:\BaiduNetdiskDownload\数据集\练手数据集\train\ants_image\0013035.jpg"
img = Image.open(img_path)
# 类型不符合
print(type(img))
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)
writer.add_image("test", img_array,dataformats='HWC')


for i in range(100):
    writer.add_scalar('y=2x',3*i,i)

writer.close()











