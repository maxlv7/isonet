import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from utils import normalize, get_label, img_2_patches, data_augmentation


def gen_data(args):
    """
    生成训练测试数据集
    """
    data_path = args.data_path
    save_path = args.save_path
    train = args.train
    test = args.test
    size = args.size
    stride = args.stride
    aug_times = args.aug_times
    gray_mode = args.gray_mode
    pic_type = args.pic_type

    train_path = Path(data_path).joinpath("Train")
    val_data_path = Path(data_path).joinpath("Test")

    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir()

    files_train = {}
    files_test = {}
    for x in train_path.glob("*"):
        if x.is_dir():
            file_list_train = [str(f_train.absolute().resolve()) for f_train in x.glob(f"*.{pic_type}")]
            files_train[x.name] = []
            files_train[x.name].extend(file_list_train)

    for y in val_data_path.glob("*"):
        if y.is_dir():
            file_list_test = [str(f_test.absolute().resolve()) for f_test in y.glob(f"*.{pic_type}")]
            files_test[y.name] = []
            files_test[y.name].extend(file_list_test)

    if gray_mode:
        train_h5 = 'train_gray.h5'
        train_h5_label = 'train_gray_label.h5'
        val_h5 = 'val_gray.h5'
        val_h5_label = 'val_gray_label.h5'
    else:
        train_h5 = 'train_rgb.h5'
        train_h5_label = 'train_rgb_label.h5'
        val_h5 = 'val_rgb.h5'
        val_h5_label = 'val_rgb_label.h5'

    if train:
        # 读取训练图片，并成生数据集
        f_train = h5py.File(save_path.joinpath(train_h5), 'w')
        f_train_label = h5py.File(save_path.joinpath(train_h5_label), 'w')

        train_num = 0
        # k->label v->filename list
        for k, v in files_train.items():
            print(k)
            print(v)
            if len(v) == 0:
                continue
            # 读取每一张大图
            for f in v:
                if gray_mode:
                    # H * W * C
                    t_pic = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                else:
                    t_pic = cv2.imread(f, cv2.IMREAD_COLOR)

                # BRG -> RGB
                t_pic = t_pic[:, :, ::-1]
                # HWC -> CHW
                t_pic = np.transpose(t_pic, (2, 0, 1))

                t_pic = normalize(t_pic)
                # CHW * patch_size
                patches = img_2_patches(t_pic, size, stride)

                # 控制样本数量
                patches = patches[:, :, :, :2400]
                # 处理每一张小图
                print(f"训练文件:{f} --> ##{patches.shape[3]}##样本")
                for nx in range(patches.shape[3]):
                    data = patches[:, :, :, nx]
                    f_train.create_dataset(str(train_num), data=data)
                    f_train_label.create_dataset(str(train_num), data=np.array(get_label(int(k))))
                    train_num += 1
                    # 数据增广
                    for mx in range(aug_times):
                        data_aug = data_augmentation(patches[:, :, :, nx].copy(), np.random.randint(1, 8))
                        f_train.create_dataset(str(train_num), data=data_aug)
                        f_train_label.create_dataset(str(train_num), data=np.array(get_label(int(k))))
                        train_num += 1

        f_train.close()
        f_train_label.close()
        print(f"训练集图片数量:{train_num}")
    if test:
        # Gen Test Data
        f_test = h5py.File(save_path.joinpath(val_h5), 'w')
        f_test_label = h5py.File(save_path.joinpath(val_h5_label), 'w')
        # k->label v->filename list
        val_num = 0
        for k, v in files_test.items():
            print(k)
            print(v)
            if len(v) == 0:
                continue
            # 读取每一张大图
            for f in v:
                if gray_mode:
                    # H * W * C
                    t_pic = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                else:
                    t_pic = cv2.imread(f, cv2.IMREAD_COLOR)

                # BRG -> RGB
                t_pic = t_pic[:, :, ::-1]
                # HWC -> CHW
                t_pic = np.transpose(t_pic, (2, 0, 1))

                t_pic = normalize(t_pic)
                # CHW * patch_size
                patches = img_2_patches(t_pic, size, stride)

                # 处理每一张小图
                print(f"测试文件:{f} --> ##{patches.shape[3]}##样本")
                for nx in range(patches.shape[3]):
                    data = patches[:, :, :, nx]
                    f_test.create_dataset(str(val_num), data=data)
                    f_test_label.create_dataset(str(val_num), data=np.array(get_label(int(k))))
                    val_num += 1

        f_test.close()
        f_test_label.close()
        print(f"测试集图片数量:{val_num}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True, help="是否想要生成训练集数据")
    parser.add_argument("--test", type=bool, default=True, help="是否想要生成测试集数据")
    parser.add_argument("--stride", type=int, default=64, help="截取图片的步长")
    parser.add_argument("--aug_times", type=int, default=0, help="数据增广的次数")
    parser.add_argument("--gray_mode", type=bool, default=False, help="使用灰度图训练模型")
    parser.add_argument("--size", type=int, default=64, help="截取的图片大小")
    parser.add_argument("--pic_type", type=str, default="tif", help="数据集中图片的后缀名")
    parser.add_argument("--data_path", type=str, default="isonet_tif", help="训练数据集文件夹的的位置")
    parser.add_argument("--save_path", type=str, default="data_64_64_aug3",
                        help="生成的数据集存储的位置")

    args = parser.parse_args()
    # 输出参数
    for p, v in args.__dict__.items():
        print('\t{}: {}'.format(p, v))

    gen_data(args)
