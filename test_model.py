from pathlib import Path
from torchvision.models import resnet18
import cv2
import torch
import math
from isonet import ISONet
from utils import img_2_patches, get_label

net = ISONet()

# 加载模型
# if torch.cuda.is_available():
#     net.load_state_dict(torch.load("net.pth"))
# model_path = "models/checkpoint/net_1.cpth"
path = Path("models/checkpoint")

for n in path.glob("*.cpth"):
    model_path = n
    print(model_path)
    checkpoint = True
    if checkpoint:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["net"])
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 读入一张图片
    # BGR -> RGB
    img = cv2.imread(r"data/Test/100/0108.tif")[:, :, ::-1]
    # HWC ->CHW
    img = img.transpose(2, 0, 1)

    # normalize
    img = img / 255.
    # 把图片分割为若干小块
    patches = img_2_patches(img, 64, 64)
    print(f"patches {patches.shape[3]}")
    correct100 = 0
    correct200 = 0
    correct400 = 0
    correct800 = 0
    correct1600 = 0
    correct3200 = 0
    fail = 0

    for nx in range(patches.shape[3]):
        with torch.no_grad():
            p = torch.from_numpy(patches[:, :, :, nx]).to(dtype=torch.float32).unsqueeze(0)
            pre = net(p)

            real = math.pow(2, pre.item()) * 100
            # print(real)
            if pre.item()<=0:
                fail+=1
            if 80 <= real <= 120:
                correct100 += 1
            if 180 <= real <= 220:
                correct200 += 1
            if 380 <= real <= 420:
                correct400 += 1
            if 780 <= real <= 820:
                correct800 += 1
            if 1580 <= real <= 1620:
                correct1600 += 1
            if 3180 <= real <= 3220:
                correct3200 += 1


    ps = patches.shape[3]
    print(f"iso100: {correct100}/{ps} [{correct100/ps:.4f}]")
    print(f"iso200: {correct200}/{ps} [{correct200/ps:.4f}]")
    print(f"iso400: {correct400}/{ps} [{correct400/ps:.4f}]")
    print(f"iso800: {correct800}/{ps} [{correct800/ps:.4f}]")
    print(f"iso1600: {correct1600}/{ps} [{correct1600/ps:.4f}]")
    print(f"iso3200: {correct3200}/{ps} [{correct3200/ps:.4f}]")
    print(f"fail: {fail}/{ps} [{fail/ps:.4f}]")