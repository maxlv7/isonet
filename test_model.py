import cv2
import torch

from ISONet import ISONet
from utils import img_2_patches,get_label

net = ISONet()

# 加载模型
net.load_state_dict(torch.load("net.pth", map_location=torch.device('cpu')))

# 类别
label = {
    "100": 0,
    "200": 1,
    "400": 2,
    "800": 3,
    "1600": 4,
    "3200": 5
}

# 读入一张图片
# BGR -> RGB
img = cv2.imread(r"3480.tif")[:, :, ::-1]
# HWC ->CHW
img = img.transpose(2, 0, 1)

# normalize
img = img / 255.

# 把图片分割为若干小块
patches = img_2_patches(img, 64, 64)
print(f"patches {patches.shape[3]}")
correct = 0
for nx in range(patches.shape[3]):
    with torch.no_grad():
        p = torch.from_numpy(patches[:, :, :, nx]).to(dtype=torch.float32).unsqueeze(0)
        pre = net(p)
        # 这里判断类别
        classes = pre.argmax().item()
        # print(classes)
        if classes == get_label(800):
            correct += 1

print(f"{correct}/{patches.shape[3]}")
print(correct / patches.shape[3])
