from typing import Sequence
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import RandomSampler, BatchSampler

class ChatDataSet(Dataset):
    def __init__(self, data, tokenizer) -> None:
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> T_co:
        source, target = self.data[index]
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        return source, target

    def __len__(self):
        return len(self.data)

class SampledDataLoader(BatchSampler):
    def __init__(self, data: Dataset, batch_size: int, padding: int):
        super().__init__(RandomSampler(data), batch_size, True)
        self.padding = padding
        self.count = 0

    def __iter__(self):
        source_list = []
        target_list = []
        for i in self.sampler:
            self.count += 1
            source, target = self.sampler.data_source[i]
            source_list.append(source)
            target_list.append(target)
            if self.count % self.batch_size == 0: # batch_size is from Batchsampler super class
                assert len(source_list) == self.batch_size
                source = self._pad_sequence(source_list, padding=self.padding)
                source_list.clear()
                target = self._pad_sequence(target_list, padding=self.padding)
                target_list.clear()
                yield source, target


    def _pad_sequence(self, batch: torch.Tensor, padding:int):
        max_size = batch[0].size()
        max_len = max([sequence.size(0) for sequence in batch])
        out_dim = (len(batch), max_len) + max_size[1:]
        padded_sequence = batch[0].new_full(out_dim, padding)
        for i, sequence in enumerate(batch):
            length = sequence.size(0)
            padded_sequence[i, :length, ...] = sequence
        return padded_sequence
