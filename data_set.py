import random
from pathlib import Path
from sys import exit

import h5py
import numpy as np
import torch
import torch.utils.data as dt


class ISONetData(dt.Dataset):

    def __init__(self, data_path=None, train=True, gray_mode=False, shuffle=False) -> None:
        if data_path is not None:
            data_path = Path(data_path)
            if not data_path.exists():
                print("data path not exist!")
                exit(0)

        self.train = train
        self.gray_mode = gray_mode
        if self.gray_mode:
            self.train_h5 = 'train_gray.h5'
            self.train_h5_label = 'train_gray_label.h5'
            self.val_h5 = 'val_gray.h5'
            self.val_h5_label = 'val_gray_label.h5'
        else:
            self.train_h5 = 'train_rgb.h5'
            self.train_h5_label = 'train_rgb_label.h5'
            self.val_h5 = 'val_rgb.h5'
            self.val_h5_label = 'val_rgb_label.h5'

        if self.train:
            self.h5 = h5py.File(data_path.joinpath(self.train_h5), 'r')
            self.h5_label = h5py.File(data_path.joinpath(self.train_h5_label), 'r')
        else:
            self.h5 = h5py.File(data_path.joinpath(self.val_h5), 'r')
            self.h5_label = h5py.File(data_path.joinpath(self.val_h5_label), 'r')

        self.keys = list(self.h5.keys())
        if shuffle:
            random.shuffle(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        data = torch.from_numpy(np.array(self.h5.get(str(index)))).to(dtype=torch.float32)
        label = torch.from_numpy(np.array(self.h5_label.get(str(index)))).to(dtype=torch.float32)
        return data, label


if __name__ == '__main__':
    data = ISONetData(data_path="ttt", train=False)
