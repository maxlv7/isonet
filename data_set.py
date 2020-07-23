import random
from pathlib import Path
from sys import exit

import h5py
import numpy as np
import torch
import torch.utils.data as dt
from torch.utils.data.dataloader import DataLoader


class ISONetData(dt.Dataset):

    def __init__(self, data_path=None, train=True, gray_mode=False, shuffle=False) -> None:
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

        self.keys = list(self.h5.keys())
        if shuffle:
            random.shuffle(self.keys)
        self.len = self.h5.__len__()
        self.h5.close()
        self.h5_label.close()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        if self.train:
            h5 = h5py.File(self.data_path.joinpath(self.train_h5), 'r')
            h5_label = h5py.File(self.data_path.joinpath(self.train_h5_label), 'r')
        else:
            h5 = h5py.File(self.data_path.joinpath(self.val_h5), 'r')
            h5_label = h5py.File(self.data_path.joinpath(self.val_h5_label), 'r')
        data = torch.from_numpy(np.array(h5[str(index)])).to(dtype=torch.float32)
        # print(f">>> index:[{index}],label:{np.array(h5_label[str(index)])}")
        label = torch.from_numpy(np.array(h5_label[str(index)])).to(dtype=torch.float32)
        h5.close()
        h5_label.close()
        return data, label

    def __del__(self):
        print("del.........")


if __name__ == '__main__':

    dataset = ISONetData(data_path="data_64_32")
    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    for i, (data, label) in enumerate(data_loader, 0):
        print(i)
