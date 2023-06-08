import torch
from torch import Tensor
from typing import List
import random


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, timestamps: List, device: torch.device, train_mode: bool):
        self.train_mode = train_mode
        self.device = device
        if self.train_mode is True:
            curr_timestamps = timestamps[:-1]
            next_timestamps = timestamps[1:]
            self.dataset = [
                (cur_t, next_t)
                for (cur_t, next_t) in zip(curr_timestamps, next_timestamps)
            ]
            self.shuffle_()
        else:
            self.dataset = timestamps

    def shuffle_(self):
        """当需要进行数据集shuffle时，必须手动调用这个方法，pytorch DataLoader自带的shuffle参数失效"""
        if self.train_mode is not True:
            NotImplementedError(
                "Please do not shuffle dataset when tesing, otherwise the game.core module will not perform as expect"
            )
        random.shuffle(self.dataset)

    def __iter__(self):
        for ts in self.dataset:
            yield ts
