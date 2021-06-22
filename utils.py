import torch
import numpy as np

import config as config

def subsequent_mask(size):
    shape = (1, size, size)
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8)
    return torch.from_numpy(mask) == 0

# mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# return mask


def create_masks(source, target, pad=0):
    source_mask = (source != pad).to(config.device)

    target_mask = (target != pad).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask)

    return source_mask, target_mask

def seed_everything(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True