from typing import Sequence
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import RandomSampler, BatchSampler

class ChatDataSet(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        self.size = len(data)

    def __getitem__(self, index) -> T_co:
        source = torch.LongTensor(self.data[index][0])
        target = torch.LongTensor(self.data[index][1])
        return source, target

    def __len__(self):
        return self.size
