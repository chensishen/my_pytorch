import torchvision
from PIL import Image
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_path = r"./cat.jpg"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                             torchvision.transforms.ToTensor(),])
image = transforms(image)
print(image.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("./my_train_model_pth/my_train_model99.pth",weights_only=False)
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image.to(device))
print(output)

print(output.argmax(dim=1))




