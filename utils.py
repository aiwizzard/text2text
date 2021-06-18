import torch
import numpy as np
import pickle
from tqdm import tqdm

import config as config

def subsequent_mask(size):
    shape = (1, size, size)
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8)
    return torch.from_numpy(mask) == 0

def create_masks(source, target, pad=0):
    source_mask = (source != pad).to(config.device)

    target_mask = (target != pad).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask)

    return source_mask, target_mask

def create_train_data(file_path: str, tokenizer, use_pickle=False) -> list:
    data = []
    if use_pickle:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in tqdm(range(0, len(lines), 3)):
            data.append(tuple(map(tokenizer.convert, lines[i: i+2])))
        with open('.data/train_data.pkl', 'wb') as file:
            pickle.dump(data, file)
    return data
