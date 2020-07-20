from pathlib import Path

import cv2
import h5py
import numpy as np

from utils import normalize, get_label, img_2_patches


def gen_data(data_path, save_path=None, train=True, test=False, size=64, stride=64, max_patches=None, aug_times=1,
             gray_mode=False):
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
            file_list_train = [str(f_train.absolute().resolve()) for f_train in x.glob("*.tif")]
            files_train[x.name] = []
            files_train[x.name].extend(file_list_train)

    for y in val_data_path.glob("*"):
        if y.is_dir():
            file_list_test = [str(f_test.absolute().resolve()) for f_test in y.glob("*.tif")]
            files_test[y.name] = []
            files_test[y.name].extend(file_list_test)

    # print(files_train)
    # print(files_test)

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

                # 处理每一张小图
                print(f"Train file:{f} --> ##{patches.shape[3]}##samples")
                for nx in range(patches.shape[3]):
                    # data = data_augmentation(patches[:, :, :, nx].copy(), np.random.randint(0, 7))
                    data = patches[:, :, :, nx]
                    f_train.create_dataset(str(train_num), data=data)
                    f_train_label.create_dataset(str(train_num), data=np.array(get_label(int(k))))
                    train_num += 1

        f_train.close()
        f_train_label.close()
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
                print(f"Test file:{f} --> ##{patches.shape[3]}##samples")
                for nx in range(patches.shape[3]):
                    # data = data_augmentation(patches[:, :, :, nx].copy(), np.random.randint(0, 7))
                    data = patches[:, :, :, nx]
                    f_test.create_dataset(str(val_num), data=data)
                    f_test_label.create_dataset(str(val_num), data=np.array(get_label(int(k))))
                    val_num += 1

        f_test.close()
        f_test_label.close()


if __name__ == '__main__':
    gen_data(data_path="data", save_path="data_64_32", train=True, test=True, stride=64)
