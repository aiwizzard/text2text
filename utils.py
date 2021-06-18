import torch
import numpy as np

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

