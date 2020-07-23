import cv2
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from isonet import ISONet
from utils import img_2_patches, normalize

net = ISONet()

# 加载模型

model_path = "models/net_best_2020-07-21_01:08:34.pth"
# model_path = "models/net_best_mse_mean.pth"
print(model_path)

checkpoint = False
if checkpoint:
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["net"])
else:
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 读入一张图片
# BGR -> RGB
img = cv2.imread(r"data/Test/800/1338_800.tif")[:, :, ::-1]
# HWC ->CHW
img = img.transpose(2, 0, 1)

# normalize
img = normalize(img)
# 把图片分割为若干小块
patches = img_2_patches(img, 64, 64)
c, h, w = img.shape
Hb = int(np.floor(h / 64))
Wb = int(np.floor(w / 64))

print(f"patches {patches.shape[3]}")
x = np.linspace(1, patches.shape[3], patches.shape[3])
y = []
res = []
for nx in range(patches.shape[3]):
    with torch.no_grad():
        p = torch.from_numpy(patches[:, :, :, nx]).to(dtype=torch.float32).unsqueeze(0)
        pre = net(p)

        value = pre.item()
        res.append(value)
        y.append(value)
        predict_iso = math.pow(2, pre.item()) * 100
        print(predict_iso)

y = np.array(y)

# plot scatter
plt.scatter(x,y)
plt.xlabel('index')
plt.ylabel('pre_iso')
plt.show()

# plot
res = np.array(res)
plt.imshow(res.reshape([Hb, Wb]))
plt.colorbar()
plt.show()