# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/9
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader


class test_dataset(Dataset):
    def __init__(self):
        self.x = list(range(10))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


torch.random.manual_seed(1234)
test_dataloader = DataLoader(
        test_dataset(),
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True
)  # sampler

i = 0
for data in test_dataloader:
    print(data)
    i = i + 1
    if i == 3:
        break

i = 0
for data in test_dataloader:
    print(data)
    i = i + 1
    if i == 3:
        break
