import argparse
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from isonet import ISONet
from utils import img_2_patches, normalize


def test_model(args):
    net = ISONet()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    net = net.to(device=device)
    model_path = args.model_path
    checkpoint = args.checkpoint
    print(f"load model:{model_path}")
    if checkpoint:
        # 加载模型
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device))["net"])
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # 读入测试图片,BGR -> RGB
    img = cv2.imread(args.pic_path)[:, :, ::-1]
    # HWC ->CHW
    img = img.transpose(2, 0, 1)
    # 归一化
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
    start_time = time.time()
    for nx in range(patches.shape[3]):
        with torch.no_grad():
            p = torch.from_numpy(patches[:, :, :, nx]).to(dtype=torch.float32,device=device).unsqueeze(0)
            pre = net(p)

            value = pre.item()
            res.append(value)
            y.append(value)
            predict_iso = math.pow(2, pre.item()) * 100
            # print(predict_iso)

    y = np.array(y)
    end_time = time.time()
    print(f"使用时间：{(end_time-start_time):.2f}s")
    # plot scatter
    plt.scatter(x, y)
    plt.xlabel('index')
    plt.ylabel('predict_iso')
    plt.show()

    # plot
    res = np.array(res)
    plt.imshow(res.reshape([Hb, Wb]))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="模型的位置")
    parser.add_argument("--pic_path", type=str, help="测试图片的位置")
    parser.add_argument("--checkpoint", action='store_true', help="是否加载的是checkpoint")

    args = parser.parse_args()
    args.model_path = 'models/net2_best_64aug3_jpeg.pth'
    args.pic_path = 'data_jpg/Test/200/Nikon_D200_0_14902.JPG'
    # 输出参数
    for p, v in args.__dict__.items():
        print('\t{}: {}'.format(p, v))
    test_model(args=args)
