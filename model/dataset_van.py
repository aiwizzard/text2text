import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import RandomSampler, BatchSampler
import torch.nn.utils.rnn as rnn

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
                source = rnn.pad_sequence(source_list, batch_first=True, padding_value=self.padding)
                source_list.clear()
                target = rnn.pad_sequence(target_list, batch_first=True, padding_value=self.padding)
                target_list.clear()
                yield source, target

