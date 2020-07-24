import random
from pathlib import Path
from sys import exit

import h5py
import numpy as np
import torch
import torch.utils.data as dt
from torch.utils.data.dataloader import DataLoader


class ISONetData(dt.Dataset):

    def __init__(self, data_path=None, train=True, gray_mode=False) -> None:
        print("init...")
        if data_path is not None:
            self.data_path = Path(data_path)
            if not self.data_path.exists():
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
            self.h5 = h5py.File(self.data_path.joinpath(self.train_h5), 'r')
            self.h5_label = h5py.File(self.data_path.joinpath(self.train_h5_label), 'r')
        else:
            self.h5 = h5py.File(self.data_path.joinpath(self.val_h5), 'r')
            self.h5_label = h5py.File(self.data_path.joinpath(self.val_h5_label), 'r')

        self.len = self.h5.__len__()
        self.h5.close()
        self.h5_label.close()
        self.h5 = None
        self.h5_label = None

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        # h5py with pytorch dataloader num_workers>0 bug
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/26
        if self.h5 is None and self.h5_label is None:
            if self.train:
                # open h5py file should not in __init__
                self.h5 = h5py.File(self.data_path.joinpath(self.train_h5), 'r', swmr=True)
                self.h5_label = h5py.File(self.data_path.joinpath(self.train_h5_label), 'r', swmr=True)
            else:
                self.h5 = h5py.File(self.data_path.joinpath(self.val_h5), 'r', swmr=True)
                self.h5_label = h5py.File(self.data_path.joinpath(self.val_h5_label), 'r', swmr=True)

        data = torch.from_numpy(np.array(self.h5[str(index)])).to(dtype=torch.float32)
        label = torch.from_numpy(np.array(self.h5_label[str(index)])).to(dtype=torch.float32)
        return data, label


if __name__ == '__main__':

    dataset = ISONetData(data_path="data_64_32")
    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    for i, (data, label) in enumerate(data_loader, 0):
        print(i)
